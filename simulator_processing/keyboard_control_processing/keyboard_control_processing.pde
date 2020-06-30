import java.io.IOException;

// keyboard variables
//boolean [] keys;
boolean keys[] = new boolean[4]; // for storing whether a key is pressed or not

void setup()
 {
   size(200, 200);
   background(255);
   
   fill(0);
 }
 void draw() 
 {/*
   if ( keys[0]) 
   {  
     text("1", 50, 50);
   }
   if ( keys[1]) 
   {
     text("2", 100, 100);
   }
   if(keys[0] && keys[1]){
     text("3", 150,150);
   }*/
   print(keys[0]);print(keys[1]);print(keys[2]);print(keys[3]);
   println("");
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
