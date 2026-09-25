# Dynamic Privacy Shield

Real-time visual privacy protection for desktop screens, combining gaze-contingent (foveated) blurring with shoulder-surfing detection.

Developed as part of the diploma thesis _Development of a visual privacy protection system through dynamic gaze tracking and shoulder surfing detection_, Department of Computer Engineering and Informatics, University of Patras.

---

## Overview

Dynamic Privacy Shield keeps legible only the region of the screen the user is currently looking at and blurs the rest. In parallel, it monitors the camera feed for faces. When a face that does not match an enrolled user appears (a potential shoulder surfer), the system switches to complete concealment of the screen content (Panic Mode).

The system is built from three components:

- **Gaze tracking.** Estimates the user's point of regard on screen through one of three interchangeable backends.
- **Foveated overlay.** A screen overlay that blurs everything except a clear region around the current gaze point.
- **Face detection and recognition.** YuNet (detection) and SFace (recognition) distinguish enrolled users from other people in the camera's field of view.

## Features

- Three gaze backends, selectable at start-up:

  | Backend              | Hardware                      | Interface                                                                          |
  | -------------------- | ----------------------------- | ---------------------------------------------------------------------------------- |
  | Webcam               | Standard webcam               | MediaPipe FaceMesh (iris landmarks)                                                |
  | Tobii Eye Tracker 4C | Tobii 4C                      | Tobii Stream Engine SDK, through a C++ subprocess bridge (`tobii_gaze_bridge.exe`) |
  | Tobii Pro Spectrum   | Tobii Pro Spectrum (Ethernet) | Tobii Pro SDK (`tobii_research`)                                                   |

- Face enrollment of authorized users and gaze calibration.
- Four-state privacy state machine: `NO_FACE → CLEAR → PANIC → HARD_LOCK`.
- `PANIC` is held for 4 seconds after the last detection of an unauthorized face, so that brief detection dropouts do not expose the screen.
- `HARD_LOCK` is fail-secure and has no bypass mechanism.
- System tray resident mode.
- Experiment logging: one `.xlsx` workbook per session (Metadata and Data sheets), organized in per-backend subfolders, plus session video recording with a burned-in timestamp overlay.

## Architecture

- **Privacy state machine.** All protection decisions are governed by the four states listed above. [TODO: one line per state describing when it is entered and what the screen shows.]
- **Two-thread architecture.** [TODO: which work runs on each thread.]
- **Gaze backend abstraction.** Backends implement a common interface (Strategy pattern), so the rest of the application is independent of the tracker in use.
- **Foveated rendering.** The clear region is cut out of the blur layer using Qt's `QPainter::CompositionMode_DestinationOut`.
- **Face pipeline.** YuNet detects all faces in the frame. SFace computes an embedding for each face, which is compared with the enrolled embeddings by cosine similarity (match threshold: 0.38).

## Requirements

- Windows (the system was developed and tested on Windows).
- Python [TODO: version]. Check compatibility with `tobii-research` before choosing a version.
- A webcam. Face detection relies on the camera feed regardless of the selected gaze backend.
- Optional: Tobii Eye Tracker 4C and/or Tobii Pro Spectrum.

## Installation

```powershell
git clone https://github.com/[TODO: username]/[TODO: repository].git
cd [TODO: repository]
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Face models

The YuNet and SFace ONNX models used by the system are included in the repository root:

- `face_detection_yunet_2023mar.onnx` (YuNet face detector)
- `face_recognition_sface_2021dec.onnx` (SFace face recognizer)

Both were obtained from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo); see the OpenCV Zoo for their licenses. If you download them again, check the file size against the one listed there. An interrupted download leaves a truncated file that fails to load.

### Tobii Eye Tracker 4C (optional)

The bridge links against the Tobii Stream Engine SDK. The SDK is distributed by Tobii under its Software Development License Agreement (SDLA) and is **not** included in this repository.

1. Install the Tobii 4C runtime software and make sure the tracker is working.
2. Obtain Tobii Stream Engine SDK v2.2.2.363 from Tobii.
3. Build the bridge from the sources in `tobii_bridge/`: [TODO: build steps: compiler/IDE, include and library paths].
4. By default, `main.py` expects the executable at `tobii_bridge/build/Release/tobii_gaze_bridge.exe`. A different path can be given as a command-line option (see `python main.py --help`). The SDK runtime DLL must be available to the executable at runtime (e.g., in the same directory).

The bridge targets the older Stream Engine API, in which `tobii_device_create` takes three parameters. Newer SDK versions require changes to the bridge source.

### Tobii Pro Spectrum (optional)

1. Install Tobii Pro Eye Tracker Manager and connect the tracker via Ethernet.
2. Confirm that the tracker is detected in Eye Tracker Manager before launching the application.
3. The Python SDK (`tobii-research`) is installed through `requirements.txt`.

## Usage

```powershell
python main.py
```

1. Select a gaze backend in the start-up dialog.
2. Enroll your face. Enrollment data are stored locally (`trusted_faces.npz`, `trusted_thumbnails/`).
3. Complete the gaze calibration.
4. The foveated overlay starts, and the application keeps running from the system tray.

[TODO: how to stop the application; keyboard shortcuts, if any.]

## Building a standalone executable

```powershell
pyinstaller [TODO: name].spec
```

Use the `.spec` file included in the repository. MediaPipe loads `.tflite` and `.binarypb` data files at runtime, and PyInstaller does not collect them automatically. The spec file gathers them with `os.walk()`, so a build without it will fail at runtime.

## Data and privacy

This repository contains **no participant data, no face images and no face embeddings**. The following files and directories are created locally at runtime and are excluded through `.gitignore`:

| Path                    | Contents                                                               |
| ----------------------- | ---------------------------------------------------------------------- |
| `experiment_logs/`      | Session logs and recordings (personal data of experiment participants) |
| `trusted_thumbnails/`   | Face images of enrolled users (biometric data)                         |
| `trusted_faces.npz`     | Face embeddings of enrolled users (biometric data)                     |
| `gaze_calibration.json` | User-specific gaze calibration                                         |

## Known limitations

- The Tobii Stream Engine license (SDLA, Section 3.6) prohibits storing gaze data for analytical purposes. This constrains what can be logged with the Tobii 4C backend.
- Tobii Pro Lab and the Tobii 4C software cannot run concurrently on the same machine.
- The Tobii 4C bridge is Windows-only.

## Citation

If you use this work, please cite the thesis:

```bibtex
@mastersthesis{tsaroucha2026privacyshield,
  author = {Tsaroucha, Eleni},
  title  = {Development of a visual privacy protection system through dynamic gaze tracking and shoulder surfing detection},
  school = {Department of Computer Engineering and Informatics, University of Patras},
  type   = {Diploma thesis},
  year   = {2026}
}
```

## References

- Wu, W., Peng, H., & Yu, S. (2023). YuNet: A tiny millisecond-level face detector. _Machine Intelligence Research_.
- Zhong, Y., Deng, W., Hu, J., Zhao, D., Li, X., & Wen, D. (2021). SFace: Sigmoid-constrained hypersphere loss for robust face recognition. _IEEE Transactions on Image Processing_.
- Lugaresi, C., et al. (2019). MediaPipe: A framework for building perception pipelines. _arXiv:1906.08172_.

## Author

Eleni Tsaroucha, Department of Computer Engineering and Informatics, University of Patras.
