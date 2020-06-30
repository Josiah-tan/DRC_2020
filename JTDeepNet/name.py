
#================================================================
#
#   File name   : name.py
#   Author      : Josiah Tan
#   Created date: 23/06/2020
#   Description : Contains name management class for objects stored in a dictionary
#
#================================================================

#================================================================

class NameManager:
    num_instances_name = {}
       
    @classmethod
    def change_name(cls, name):
        """Changes the name of the instance which will be recorded in the JTDNN instance dictionary"""
        if name != "":
            cls.name = name
            if name not in cls.num_instances_name:
                cls.num_instances_name[cls.name] = 0
                
        cls.num_instances_name[cls.name] += 1
        
    @classmethod    
    def add_name(cls, self):
        """adds the instance and its corresponding name to the JTDNN instance dictionary"""
        cls_obj_key =  f"{cls.name}{cls.num_instances_name[cls.name]}"
        self.key_name = cls_obj_key
        self.jtdnn_obj.graph_lis.append(cls_obj_key)
        self.jtdnn_obj.graph_dict[cls_obj_key] = self
        
        