# bytes-to-ndarray

Run the following:

```
python3 -m venv venv
source venv/bin/activate
python3 -m pip install numpy meson meson-python
meson setup build
meson compile -C build
PYTHONPATH=$PWD/build python createarray.py
``