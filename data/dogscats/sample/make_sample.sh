#!/bin/bash

##########################
#  Prepares Sample Data  #
##########################


train_size=$1
valid_size=$2
test_size=$3


# Create Directory Structure
mkdir train
cd train/
mkdir dogs/
mkdir cats/
cd ..

mkdir test
cd test/
mkdir subdir_for_keras_ImageDataGenerator
cd ..

mkdir valid
cd valid/
mkdir dogs/
mkdir cats/
cd ..

# Make the sample directories
# Using the real data files

## Sample Train
cd ../train/

cd dogs/
for file in $(ls -p | grep -v / | tail -$train_size)
do
	cp $file ../../sample/train/dogs/
done
cd ..

cd cats/
for file in $(ls -p | grep -v / | tail -$train_size)
do
	cp $file ../../sample/train/cats/
done
cd ..

cd ../sample/

## Sample Valid
cd ../valid/

cd dogs/
for file in $(ls -p | grep -v / | tail -$valid_size)
do
	cp $file ../../sample/valid/dogs/
done
cd ..

cd cats/
for file in $(ls -p | grep -v / | tail -$valid_size)
do
	cp $file ../../sample/valid/cats/
done
cd ..

cd ../sample/

## Sample Test
cd ../test/

cd subdir_for_keras_ImageDataGenerator/
for file in $(ls -p | grep -v / | tail -$test_size)
do
	cp $file ../../sample/test/subdir_for_keras_ImageDataGenerator/
done
cd ..

cd ../sample/

