String cvtArray2Str( boolean [] keys){
  String build_str = "";
  for (boolean i : keys){
    if (i){
      build_str = build_str + ' ' + '1';
    }
    else{
      build_str = build_str + ' ' +  '0';
    }
  }
  build_str = build_str + '\n';

  return build_str;
}
