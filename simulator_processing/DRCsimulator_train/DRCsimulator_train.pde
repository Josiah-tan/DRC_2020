// for keyboard control
boolean keys[] = new boolean[4];
boolean record = false; //turn off communication with python initially
boolean quit = false; // closes the program

// create an image object
PImage img;

// Communication with Python
String str;
import processing.net.*; 
Client myClient;

// dimensions of screen viewable relative to the whole image
int view_factor_x = 10;
int view_factor_y = 10;

// view settings
int view_z = 0; // distance the image is away from us
float dist_z = -(360/2.0) / tan(PI*30.0 / 180.0); // actual distance from normal setting (view_z = 0)
int view_plane_x_angle = 50; // angle of the cartesian plane rotated about x for viewing

// initial position of where you want the m.p of the screen to be
//float initial_x = -120;
//float initial_y = -450;

float initial_x = 1680;
float initial_y = 1238;

// initial rotation settings
int initial_angle = -82;
int radius = 300; // radius of curvature
int angle_increment = 1; // degrees

// displacement increments
int disp_increment = 5;

// bias correction from tilt
float bias = abs(view_z + dist_z) * sin(radians(view_plane_x_angle));

/*variables used for backend calculations */
// calculating coordinates of vehicle on a circumference when turning relative to the left corner of the actual image
float x = initial_x;
float y = initial_y;

// angle relative to the vertical axis
float angle = initial_angle;



void setup() {
  // Images must be in the "data" directory to load correctly
  img = loadImage("extreme_track.JPG");
  //img = loadImage("difficult_track.JPG");
  //img = loadImage("drc_track.png");
  size(640, 360, P3D); // if you change any of this, remember to change stuff below
  
  myClient = new Client(this, "127.0.0.1", 5000); 
}

void draw() {
  
  background(255);
  translate(0, 0, view_z); // move the image towards the page a little
  
  //translate(64w0/2 - bias * sin(radians(angle)), 360/2 + bias * cos(radians(angle))); 
  translate(width/2, height/2); // move coordinate system to middle of page
  rotateX(radians(view_plane_x_angle)); // tilt the image like from | to /
  translate(0, bias); // changes the coordinate system so that rotation is from your POV
  rotateZ(radians(angle)); // rotation parallel to the plane?
  //rect(-x, -y, bias * 2, bias * 2);
  image(img, -x , -y , width * view_factor_x, height * view_factor_y); // plonks the image down
  
  rotating_z(keys); // changes x, y and angle depending on keys, probs should use pointers and stuff, but cbbs
  
  if (record){
    str = cvtArray2Str(keys); // convert keys to a string [0,1,0,1] => 0101
    myClient.write(str); // write the string to my python script
  }
  
  if (quit){
    exit();
  }
  
  
  //println(x, y, angle);
  //circular_motion(keys, radius, angle_increment);
  //circle(0, 0, 500);
  //print(bias);
  //translate(bias * sin(radians(angle)), -bias * cos(radians(angle)));
  //translate(bias * sin(radians(angle)), bias * cos(radians(angle)));
  //translate(bias * sin(radians(angle)), bias * cos(radians(angle)));
  //rotateZ(radians(angle));
  //circle(0, 0, bias);
  //circle(bias * sin(radians(angle)),bias * cos(radians(angle)), 10);
  //circle(0, 0, bias);
  //circle(-bias * sin(radians(angle)), -bias * cos(radians(angle)), 10);
  //image(img, -x , -y , width * view_factor_x, height * view_factor_y);
  //up_down_left_right(keys);
  //rotating_z(keys);
  //println(x, y);
  /*
  translate(width / 2, height / 2, view_z);
  rotateX(radians(view_plane_x_angle));
  translate(initial_x, initial_y);
  rotateZ(radians(initial_rotate));
  imageMode(CENTER);
  image(img, 0, 0, width * view_factor_x, height * view_factor_y);
  print(initial_x, initial_y);
  //up_down_left_right(keys);
  change_pos(keys);
  */
}
