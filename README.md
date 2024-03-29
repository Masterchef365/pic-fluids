# Particle life And FLIP Fluid simulation, with node graph
This is a combination of a [FLIP fluid solver](https://youtu.be/XmzBREkK8kY), and a [Particle Life](https://youtu.be/p4YirERTVF0) simulation. On each page refresh, fresh rules are chosen for the particle life simulation. **This simulation is interactive, drag the particles around!**

### Try the online version!
https://masterchef365.github.io/pic-fluids/

### Node graph
The node graph portion can be considered to be a type of self-modifying code; changes to the node graph re-compile a portion of the webassembly code and use it to accelerate the simulation.

### Test locally
Make sure you are using the latest version of stable rust by running `rustup update`.

`cargo run --release`

On Linux you need to first run:

`sudo apt-get install libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libxkbcommon-dev libssl-dev`

On Fedora Rawhide you need to run:

`dnf install clang clang-devel clang-tools-extra libxkbcommon-devel pkg-config openssl-devel libxcb-devel gtk3-devel atk fontconfig-devel`

### Web Deploy
Just run `trunk build --release`.

### ROADMAP:
 - [x] Add save states
 - [ ] BUG: it is not possible to add new nodes in mobile!
    * Can be fixed by adding a button which floats on top and opens the finder menu
 - [ ] Add a way to keep a certain set of "projects" around
 - [x] Add another set of nodes for per-particle alongside the existing per-neighbor. Include a dt!
 - [ ] Specify output node type!
 - [ ] Subroutines? !
 - [ ] "Maximum" and "Minimum" functions
 - [ ] Vector reduction functions
