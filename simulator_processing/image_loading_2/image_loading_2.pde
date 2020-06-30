// for keyboard control
boolean keys[] = new boolean[4];

// create an image object
PImage img;

// dimensions of screen viewable relative to the whole image
int view_factor_x = 6;
int view_factor_y = 6;

// initial position of the screen viewable
float initial_x = -300;
float initial_y = -2500;

// initial rotation settings
int initial_rotate = 0;
int radius = 400;

// displacement and angular increments
int disp_increment = 100;
int angle_increment = 10; // degrees

void setup() {
  // Images must be in the "data" directory to load correctly
  img = loadImage("drc_track.png");
  size(displayWidth, displayHeight);
}

void draw() {

  translate(initial_x, initial_y);
  
  rotate(radians(initial_rotate));
  
  image(img, 0, 0, displayWidth * view_factor_x, displayHeight * view_factor_y);
  print(initial_x, initial_y);
  println(displayWidth, displayHeight);
  change_pos(keys);
}
