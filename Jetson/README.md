# Setup Jetson
1. Hubungkan Jetson dengan kamera Realsense pada usb PORT 3.0
2. HUbungkan Jetson dengan internet melalui dongle wifi atau ethernet

# Menjalanakn sistem Deteksi Pose
1. Login dengan menggunakan password : pcu123
2. Buka terminal pada Jetson Nano
3. Masuk ke folder darknet menggunakan command   
```sh
$ cd darknet
```
5. Jalankan progrma deteksi pose dengan command
```sh
$ python3 realsense-inference.py
```
