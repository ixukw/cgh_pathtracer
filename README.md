# cgh_pathtracer

This is the code for a path tracer to compute computer generated holograms. Learn more about the project by reading the report at 272_final_report.pdf.

## Installation

Download the Odak library here: https://kaanaksit.com/odak/

Replace the render.cpp and vol_path_tracing.h files in lajolla from here: https://github.com/BachiLi/lajolla_public

To run the python files, install the necessary packages.

## Usage

To render phase images, run as instructed in the lajolla renderer.

To generate reconstructions using your phase images, run `python3 holo_script2.py`.

To generate reconstructions from the source image, run `python3 holo_script.py`
