#pragma once

class image {

private:
	unsigned int dimensionNum, width, height;
	

public:
	buffer* buffers;
	image() {

		width = 0;
		height = 0;
		buffers = NULL;
		dimensionNum = 0;
	}

	image(unsigned int width, unsigned int height) {

		dimensionNum = 3;

		this->width = width;
		this->height = height;

		buffers = new buffer[3];


	}

	image(unsigned int width, unsigned int height, unsigned int dimensionNum) {



		this->width = width;
		this->height = height;
		this->dimensionNum = dimensionNum;

		buffers = new buffer[dimensionNum];
	}


	void open(std::string fileName) {

		this->dimensionNum = 3;

		this->width = getBmpWidth(fileName);
		this->height = getBmpHeight(fileName);

		this->buffers = new buffer[3];
		for (int i = 0; i < 3; i++) {
			this->buffers[i] = buffer(width, height, i);
			buffers[i].open(fileName, i);
		}



	}

	void toBmp(std::string fileName) {

		float*** bmpBuffer = new float** [3];


		for (int i = 0; i < 3; i++) {
			bmpBuffer[i] = buffers[i].showBuffer();
		}

		writeBufferToBmp(fileName, width, height, bmpBuffer);

	}

	int getDimensionNum() {
		return dimensionNum;
	}

	int getWidth() {
		return width;
	}
	int getHeight() {
		return height;
	}

	buffer getBuffer(int index) {

		return buffers[index];

	}


};