import sys
import json
from PIL import Image
from decimal import *

def get_object_class(region, file, names):
    type = ""
    package = ""
    try:        
        type = region['region_attributes']['BIRD']
    except KeyError:
        print(sys.stderr, "bird info is missing in ", file)
        
    index = [item.lower() for item in names].index(type.lower())
    
    return index

def get_dark_annotation(region, size):
    x = region['shape_attributes']['x']
    y = region['shape_attributes']['y']
    width = region['shape_attributes']['width']
    height = region['shape_attributes']['height']

    _x      = (x+width/2) / size[0] # relative position of center x of rect
    _y      = (y+height/2) / size[1] # relative position of center y of rect
    _width  = width / size[0]
    _height = height / size[1]
    
    ret = str(_x) + " " + str(_y) + " " + str(_width) + " " + str(_height)

    return ret

def main():
    with open(sys.argv[1:][0]) as file:
        dict = json.load(file)
        
        try:        
            namesFile = sys.argv[1:][1]
            names = open(namesFile).read().split('\n')
        except IndexError:
            print(sys.stderr, "names file's missing from argument.\n\tnamesFile = sys.argv[1:][1]\nIndexError: list index out of range")

        for key in dict.keys():
            data = dict[key]

            imageName = data['filename']
            filename = imageName.rsplit('.', 1)[0]
            
            regions = data['regions']

            try:        
                img = Image.open(imageName)
                content = ""
                for region in regions:
                    obj_class = get_object_class(region, imageName, names)
                    annotation = get_dark_annotation(region, img.size)
                    content += "{} {}\n".format(obj_class, annotation)

                with open("{}.txt".format(filename), "w") as outFile:
                    outFile.write(content)
            except IOError:
                print(sys.stderr, "No such file" , imageName)

            

if __name__ == "__main__":
    main()