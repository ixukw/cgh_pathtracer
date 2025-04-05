# cgh_pathtracer

This is the code for a path tracer to compute computer generated holograms. Learn more about the project by reading the report at 272_final_report.pdf.

## Installation

Download the Odak library here and place the src in available imports. https://kaanaksit.com/odak/

Download embree and necessary packages to run lajolla from the master repository. https://github.com/BachiLi/lajolla_public

## Running

Please use the provided pyvenv.cfg file for setting up the virtual environment.

## Usage

To render phase images, run as instructed in the lajolla renderer.

To generate reconstructions using your phase images, run `python3 evaluation/src/generate_reconstruction.py`.

To generate reconstructions from the source image, run `python3 evaluation/src/generate_cgh_sgd.py`.
