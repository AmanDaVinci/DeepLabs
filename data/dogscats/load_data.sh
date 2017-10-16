#!/bin/bash

##########################
# Loads Data from Kaggle #
##########################

# Install required tools 
pip install kaggle-cli
apt install unzip

# Kaggle Credentials
username=$1
password=$2
competetion=$3
valid=$4
# Begin Download
kg download -u $username -p $password -c $competetion

# Unzip the files
unzip -q train.zip
unzip -q test.zip

# Create Directory Structure
cd train/
mkdir dogs/
mkdir cats/
cd ..

cd test/
mkdir subdir_for_keras_ImageDataGenerator
mv *.jpg subdir_for_keras_ImageDataGenerator/
cd ..

mkdir valid
cd valid/
mkdir dogs/
mkdir cats/
cd ..

# Populate the directories
cd train/
mv dog* dogs/
mv cat* cats/
cd ..

cd train/

cd dogs/
for file in $(ls -p | grep -v / | tail -$valid)
do
	mv $file ../../valid/dogs/
done
cd ..

cd cats/
for file in $(ls -p | grep -v / | tail -$valid)
do
	mv $file ../../valid/cats/
done
cd ..

cd ..