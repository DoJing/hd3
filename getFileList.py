import os
import cv2

def ListFilesToTxt(dir,file,wildcard,recursion):
    file_list=[]
    exts = wildcard.split(" ")
    files = os.listdir(dir)
    for name in files:
        fullname=os.path.join(dir,name)
        if(os.path.isdir(fullname) & recursion):
            ListFilesToTxt(fullname,file,wildcard,recursion)
        else:
            for ext in exts:
                if(name.endswith(ext)):
                    file_list.append(name)
                    break
    file_list.sort()
    resize = False
    factor = 0.5
    if(resize):
        for i in range(0,len(file_list)):
            fullname=os.path.join(dir,file_list[i])
            img=cv2.imread(fullname)
            height, width = img.shape[:2]
            size = (int(width * factor), int(height * factor))
            img=cv2.resize(img,size)
            fullname = os.path.join("/media/doing/C8BA5288BA5272C4/LINUX/pot", file_list[i])
        cv2.imwrite(fullname,img)
    for i in range(0,len(file_list)-1):
        file.write(file_list[i] + " ")
        file.write(file_list[i+1]+"\n")
def getFileList():
  dir="/media/doing/Samsung USB/flowerpot"     #文件路径
  outfile="flowerpot.txt"                     #写入的txt文件名
  wildcard = ".JPG"      #要读取的文件类型；

  file = open(outfile,"w")
  if not file:
    print ("cannot open the file %s for writing" % outfile)

  ListFilesToTxt(dir,file,wildcard, 1)

  file.close()


getFileList()
