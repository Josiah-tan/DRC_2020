void change_pos(boolean [] keys){
  if (keys[0]) initial_angle = initial_angle - angle_increment;
  if (keys[1]) initial_angle = initial_angle + angle_increment;
  if (keys[2]){    
    initial_y = initial_y + 100 * cos(radians(initial_angle));
    initial_x = initial_x + 100 * sin(radians(initial_angle));
  }
  
  if (keys[3]){
    initial_y = initial_y - 100 * cos(radians(initial_angle));
    initial_x = initial_x - 100 * sin(radians(initial_angle));
  }
}

void up_down_left_right(boolean [] keys){
  if (keys[0]){
    x = x - disp_increment;
  }
  if (keys[1]){
    x = x + disp_increment;
  }
  if (keys[2]){    
    y = y + disp_increment;
  }
  
  if (keys[3]){
    y = y - disp_increment;
  }
}

void circular_motion(boolean [] keys, float radius, float angle_increment){
  if (keys[0]){
    x = x + radius * (cos(radians(angle)) - cos(radians(angle + angle_increment)));
    y = y + radius * (sin(radians(angle)) - sin(radians(angle + angle_increment))); 
    angle = angle + angle_increment;
  }
    
  if (keys[1]){
    x = x + radius * (cos(radians(angle)) - cos(radians(angle - angle_increment)));
    y = y + radius * (sin(radians(angle)) - sin(radians(angle - angle_increment))); 
    angle = angle - angle_increment;
  }
  
  if (keys[3]){    
    y = y + disp_increment * cos(radians(angle));
    x = x + disp_increment * sin(radians(angle));
  }
  
  if (keys[2]){
    y = y - disp_increment * cos(radians(angle));
    x = x - disp_increment * sin(radians(angle));
  }
}

void rotating_z(boolean [] keys){
  if (keys[0]){
    angle = angle - angle_increment;
  }
  if (keys[1]){
    angle = angle + angle_increment;
  }
  if (keys[3]){    
    y = y + disp_increment * cos(radians(angle));
    x = x + disp_increment * sin(radians(angle));
  }
  
  if (keys[2]){
    y = y - disp_increment * cos(radians(angle));
    x = x - disp_increment * sin(radians(angle));
  }
}

void view_changing(boolean [] keys){
  // bug spotted in changing the view angle, it doesn't change it in the way I expected
  if (keys[0]){
    view_z = view_z - disp_increment;
  }
  if (keys[1]){
    view_z = view_z + disp_increment;
  }
  if (keys[2]){
    view_plane_x_angle = view_plane_x_angle + angle_increment;
  }
  if (keys[3]){
    view_plane_x_angle = view_plane_x_angle - angle_increment;
    
  }
}


void keyPressed(){
  if (key=='d')
    keys[0] = true;
  if (key=='a')
    keys[1] = true;
  if (key=='w')
    keys[2] = true;
  if (key=='s')
    keys[3] = true;
}

void keyReleased(){
  if (key=='d')
    keys[0] = false;
  if (key=='a')
    keys[1] = false;
  if (key=='w')
    keys[2] = false;
  if (key=='s')
    keys[3] = false;
}
