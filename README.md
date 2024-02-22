# bytes-to-ndarray

Run the following:

```
python3 -m venv venv
source venv/bin/activate
python3 -m pip install numpy meson meson-python
meson setup build
cd build && meson compile
python3 -c "import example; print(example.bytesarray([b'hello']))"
```