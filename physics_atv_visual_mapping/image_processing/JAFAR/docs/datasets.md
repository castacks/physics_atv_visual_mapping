To install the datasets, follow these steps:

```bash
mkdir data
cd data
```

0. Imagenet

Download imagenet and put it in the data/imagenet folder.

1. COCOStuff

Follow the instructions on the official [github](https://github.com/nightrome/cocostuff).

```bash
# 1. Download everything
mkdir COCOStuff
cd COCOStuff
wget --directory-prefix=downloads http://images.cocodataset.org/zips/train2017.zip
wget --directory-prefix=downloads http://images.cocodataset.org/zips/val2017.zip
wget --directory-prefix=downloads http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip

# 2. Unpack everything
mkdir -p dataset/images
mkdir -p dataset/annotations
unzip downloads/train2017.zip -d dataset/images/
unzip downloads/val2017.zip -d dataset/images/
unzip downloads/stuffthingmaps_trainval2017.zip -d dataset/annotations/

# https://github.com/xu-ji/IIC/blob/master/datasets/README.txt
wget https://www.robots.ox.ac.uk/~xuji/datasets/COCOStuff164kCurated.tar.gz
tar -xzf COCOStuff164kCurated.tar.gz
mv COCO/COCOStuff164k/curated ./curated
rm -r COCO
rm COCOStuff164kCurated.tar.gz
cd ../
```

2.  Cityscapes

```bash
mkdir cityscapes
cd cityscapes
# https://github.com/cemsaz/city-scapes-script
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=myusername&password=mypassword&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
```

3. VOC
```bash
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
rm VOCtrainval_11-May-2012.tar
```

4. ADE20K
```bash
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip
rm ADEChallengeData2016.zip
```


The data folder should have the following structure.
```
JAFAR
├── data
│   ├── ADEChallengeData2016
│   ├── COCOStuff
│   ├── cityscapes
│   ├── imagenet
│   ├── VOCdevkit
```
