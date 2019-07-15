import os


def ListFilesToTxt(dir,file,wildcard,recursion):
    exts = wildcard.split(" ")
    files = os.listdir(dir)
    for name in files:
        fullname=os.path.join(dir,name)
        if(os.path.isdir(fullname) & recursion):
            ListFilesToTxt(fullname,file,wildcard,recursion)
        else:
            for ext in exts:
                if(name.endswith(ext)):
                    file.write(name + "\n")
                    break

def getFileList():
  dir="/home/robot_lcl/catkin_ws/src/beginner_tutorials/robot_bag/Asus/image_Asus"     #文件路径
  outfile="binaries.txt"                     #写入的txt文件名
  wildcard = ".jpg .txt .exe .dll .lib"      #要读取的文件类型；

  file = open(outfile,"w")
  if not file:
    print ("cannot open the file %s for writing" % outfile)

  ListFilesToTxt(dir,file,wildcard, 1)

  file.close()


getFileList()
