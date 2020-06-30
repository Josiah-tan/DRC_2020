// for keyboard control
boolean keys[] = new boolean[4];
String str;
import processing.net.*; 
Client myClient;

void setup() { 
  size(200, 200); 
  /* Connect to the local machine at port 5204
   *  (or whichever port you choose to run the
   *  server on).
   * This example will not run if you haven't
   *  previously started a server on this port.
   */
  myClient = new Client(this, "127.0.0.1", 5000); 
} 

void draw() { 
  str = cvtArray2Str(keys);
  myClient.write(str); // send whatever you need to send here
} 

String cvtArray2Str( boolean [] keys){
  String build_str = "";
  for (boolean i : keys){
    if (i){
      build_str = build_str + '1';
    }
    else{
      build_str = build_str + '0';
    }
  }

  return build_str;
}
