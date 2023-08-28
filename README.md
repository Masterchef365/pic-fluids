# Particle life And FLIP Fluid simulation
This is a combination of a [FLIP fluid solver](https://youtu.be/XmzBREkK8kY), and a [Particle Life](https://youtu.be/p4YirERTVF0) simulation. On each page refresh, fresh rules are chosen for the particle life simulation. **This simulation is interactive, drag the particles around!**

### Try the online version!
https://masterchef365.github.io/pic-fluids/

### Test locally
Make sure you are using the latest version of stable rust by running `rustup update`.

`cargo run --release`

On Linux you need to first run:

`sudo apt-get install libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libxkbcommon-dev libssl-dev`

On Fedora Rawhide you need to run:

`dnf install clang clang-devel clang-tools-extra libxkbcommon-devel pkg-config openssl-devel libxcb-devel gtk3-devel atk fontconfig-devel`

### Web Deploy
Just run `trunk build --release`.
