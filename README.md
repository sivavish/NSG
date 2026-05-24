NSG Surveillance System — National Finalist, Smart India Hackathon 2025 (Top finalist among 50,000+ teams)
Python, PyTorch, YOLOv8, ByteTrack, SlowFast/X3D, OpenCV, FastAPI

• Developed an AI-powered multi-stream surveillance system capable of processing synchronized CCTV, drone, and body-camera feeds for real-time threat and activity monitoring

• Fine-tuned YOLOv8 on a custom 400-image surveillance dataset using PyTorch transfer learning from COCO pretrained weights, achieving 90% mAP across 500+ test scenarios involving multiple camera types, low-light environments, and partial occlusions

• Engineered a unified video ingestion pipeline supporting 3 concurrent live streams with synchronized timestamping and optimized frame handling for near real-time analysis

• Improved multi-object tracking reliability by integrating ByteTrack with an appearance-embedding re-identification buffer, significantly reducing identity-switching issues during crowd occlusion and fast subject movement

• Integrated SlowFast/X3D-based temporal action recognition through a FastAPI REST service capable of generating structured JSON incident alerts containing timestamps, confidence scores, and activity heatmaps with approximately 8-second end-to-end response latency

• Designed the system for scalable security and defense surveillance use cases; patent filing currently in progress
