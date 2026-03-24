use ndarray::{Array2, Array3, Axis};
use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::{c_char, c_int};
use std::path::Path;
use std::ptr::NonNull;
use std::slice;

#[repr(C)]
struct Sam3ResultRaw {
    width: i32,
    height: i32,
    count: i32,
    scores: *mut f32,
    boxes_xyxy: *mut f32,
    masks: *mut f32,
}

#[repr(C)]
struct Sam3HandleRaw {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn sam3_create(gguf_path: *const c_char, bpe_path: *const c_char, prefer_gpu: c_int) -> *mut Sam3HandleRaw;
    fn sam3_destroy(handle: *mut Sam3HandleRaw);
    fn sam3_predict(
        handle: *mut Sam3HandleRaw,
        image_path: *const c_char,
        prompt: *const c_char,
        out_result: *mut Sam3ResultRaw,
    ) -> c_int;
    fn sam3_predict_tokens(
        handle: *mut Sam3HandleRaw,
        image_path: *const c_char,
        tokens: *const i32,
        token_count: i32,
        out_result: *mut Sam3ResultRaw,
    ) -> c_int;
    fn sam3_result_free(result: *mut Sam3ResultRaw);
    fn sam3_get_last_error() -> *const c_char;
}

#[derive(Debug, Clone)]
pub struct Sam3Error {
    message: String,
}

impl Sam3Error {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for Sam3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}

impl std::error::Error for Sam3Error {}

fn last_error() -> Sam3Error {
    unsafe {
        let ptr = sam3_get_last_error();
        if ptr.is_null() {
            Sam3Error::new("sam3cpp native error")
        } else {
            Sam3Error::new(CStr::from_ptr(ptr).to_string_lossy().into_owned())
        }
    }
}

fn path_to_cstring(path: &Path) -> Result<CString, Sam3Error> {
    let s = path
        .to_str()
        .ok_or_else(|| Sam3Error::new(format!("path is not valid UTF-8: {}", path.display())))?;
    CString::new(s).map_err(|_| Sam3Error::new(format!("path contains NUL byte: {}", path.display())))
}

#[derive(Debug, Clone)]
pub struct Prediction {
    pub width: usize,
    pub height: usize,
    pub scores: Vec<f32>,
    pub boxes_xyxy: Vec<[f32; 4]>,
    pub masks: Array3<f32>,
}

impl Prediction {
    pub fn count(&self) -> usize {
        self.scores.len()
    }

    pub fn mask(&self, index: usize) -> Array2<f32> {
        self.masks.index_axis(Axis(0), index).to_owned()
    }
}

pub struct Sam3Model {
    handle: NonNull<Sam3HandleRaw>,
}

impl Sam3Model {
    pub fn new(gguf_path: impl AsRef<Path>) -> Result<Self, Sam3Error> {
        Self::with_options(gguf_path, true, None::<&Path>)
    }

    pub fn with_options(
        gguf_path: impl AsRef<Path>,
        prefer_gpu: bool,
        bpe_path: Option<impl AsRef<Path>>,
    ) -> Result<Self, Sam3Error> {
        let gguf_path = path_to_cstring(gguf_path.as_ref())?;
        let bpe_cstring = match bpe_path {
            Some(path) => Some(path_to_cstring(path.as_ref())?),
            None => None,
        };

        let handle = unsafe {
            sam3_create(
                gguf_path.as_ptr(),
                bpe_cstring
                    .as_ref()
                    .map(|s| s.as_ptr())
                    .unwrap_or(std::ptr::null()),
                if prefer_gpu { 1 } else { 0 },
            )
        };
        let handle = NonNull::new(handle).ok_or_else(last_error)?;
        Ok(Self { handle })
    }

    pub fn predict(
        &self,
        image_path: impl AsRef<Path>,
        prompt: impl AsRef<str>,
    ) -> Result<Prediction, Sam3Error> {
        let image_path = path_to_cstring(image_path.as_ref())?;
        let prompt = CString::new(prompt.as_ref())
            .map_err(|_| Sam3Error::new("prompt contains NUL byte"))?;
        let mut raw = Sam3ResultRaw {
            width: 0,
            height: 0,
            count: 0,
            scores: std::ptr::null_mut(),
            boxes_xyxy: std::ptr::null_mut(),
            masks: std::ptr::null_mut(),
        };

        let rc = unsafe {
            sam3_predict(
                self.handle.as_ptr(),
                image_path.as_ptr(),
                prompt.as_ptr(),
                &mut raw,
            )
        };
        if rc != 0 {
            unsafe { sam3_result_free(&mut raw) };
            return Err(last_error());
        }

        Ok(convert_prediction(raw))
    }

    pub fn predict_tokens(
        &self,
        image_path: impl AsRef<Path>,
        tokens: &[i32],
    ) -> Result<Prediction, Sam3Error> {
        let image_path = path_to_cstring(image_path.as_ref())?;
        let mut raw = Sam3ResultRaw {
            width: 0,
            height: 0,
            count: 0,
            scores: std::ptr::null_mut(),
            boxes_xyxy: std::ptr::null_mut(),
            masks: std::ptr::null_mut(),
        };

        let rc = unsafe {
            sam3_predict_tokens(
                self.handle.as_ptr(),
                image_path.as_ptr(),
                tokens.as_ptr(),
                tokens.len() as i32,
                &mut raw,
            )
        };
        if rc != 0 {
            unsafe { sam3_result_free(&mut raw) };
            return Err(last_error());
        }

        Ok(convert_prediction(raw))
    }
}

impl Drop for Sam3Model {
    fn drop(&mut self) {
        unsafe { sam3_destroy(self.handle.as_ptr()) };
    }
}

fn convert_prediction(mut raw: Sam3ResultRaw) -> Prediction {
    let width = raw.width.max(0) as usize;
    let height = raw.height.max(0) as usize;
    let count = raw.count.max(0) as usize;
    let mask_len = count * width * height;

    let scores = unsafe { slice::from_raw_parts(raw.scores, count) }.to_vec();
    let boxes_flat = unsafe { slice::from_raw_parts(raw.boxes_xyxy, count * 4) }.to_vec();
    let masks_flat = unsafe { slice::from_raw_parts(raw.masks, mask_len) }.to_vec();

    unsafe { sam3_result_free(&mut raw) };

    let boxes_xyxy = boxes_flat
        .chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect();
    let masks = Array3::from_shape_vec((count, height, width), masks_flat)
        .expect("native mask buffer had invalid CHW shape");

    Prediction {
        width,
        height,
        scores,
        boxes_xyxy,
        masks,
    }
}
