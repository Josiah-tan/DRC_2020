void change_pos(boolean [] keys){
  if (keys[0]) initial_rotate = initial_rotate - angle_increment;
  if (keys[1]) initial_rotate = initial_rotate + angle_increment;
  if (keys[2]){    
    initial_y = initial_y + 100 * cos(radians(initial_rotate));
    initial_x = initial_x + 100 * sin(radians(initial_rotate));
  }
  
  if (keys[3]){
    initial_y = initial_y - 100 * cos(radians(initial_rotate));
    initial_x = initial_x - 100 * cos(radians(initial_rotate));
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
