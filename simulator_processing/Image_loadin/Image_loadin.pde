PImage img;

void setup() {
  // Images must be in the "data" directory to load correctly
  img = loadImage("drc_track.png");
  size(displayWidth, displayHeight);
}

void draw() {
  image(img, 0, 0, displayWidth, displayHeight);
}
