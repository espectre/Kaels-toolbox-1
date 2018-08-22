import sys
import json


def main():
    with open(sys.argv[1],'r') as f:
        raw = json.load(f)
    with open(sys.argv[2],'w') as f:
        f.write('ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n')
        for img in raw:
            for box in raw[img]:
                f.write('{},{},{},{},{},{},{},{},0,0,0,0,0\n'.format(img,'labelx',box[-1],1,box[0],box[2],box[1],box[3]))

if __name__ == '__main__':
    main()
