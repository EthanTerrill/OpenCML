#pragma once

class image {

private:
	unsigned int dimensionNum, width, height;


public:
	buffer* buffers;

	image() {

		width = 0;
		height = 0;
		buffers = new buffer();
		dimensionNum = 0;
	}

	image(unsigned int width, unsigned int height) {

		dimensionNum = 3;

		this->width = width;
		this->height = height;

		buffers = new buffer[3];

		for (int i = 0; i < 3; i++)
			buffers[i] = buffer(width, height, i, 3);



	}

	image(unsigned int width, unsigned int height, unsigned int dimensionNum) {



		this->width = width;
		this->height = height;
		this->dimensionNum = dimensionNum;

		buffers = new buffer[dimensionNum];

		for (int i = 0; i < dimensionNum; i++)
			buffers[i] = buffer(width, height, i, dimensionNum);
	}


	void open(std::string fileName) {

		this->dimensionNum = 3;

		this->width = getBmpWidth(fileName);
		this->height = getBmpHeight(fileName);

		this->buffers = new buffer[3];
		for (int i = 0; i < 3; i++) {
			this->buffers[i] = buffer(width, height, i, 3);
			buffers[i].open(fileName, i);
		}



	}

	void toBmp(std::string fileName) {

		float*** bmpBuffer = new float** [3];


		for (int i = 0; i < 3; i++) {
			int a = (i < dimensionNum) * i + ((i >= dimensionNum) * (dimensionNum - 1));
			bmpBuffer[i] = buffers[a].showBuffer();
		}

		writeBufferToBmp(fileName, width, height, bmpBuffer);

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < width; j++) {
				delete bmpBuffer[i][j];
			}
			delete[] bmpBuffer[i];
		}
		delete[] bmpBuffer;
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

	void wipe() {

		for (int i = 0; i < dimensionNum; i++) {
			buffers[i].wipe();
		}
	}

	void clear() {

		for (int i = 0; i < dimensionNum; i++) {
			buffers[i].clear();
		}
	}
	void threshold(float threshold) {

		for (int i = 0; i < dimensionNum; i++) {
			buffers[i].threshold(threshold);
		}
	}

	buffer getBuffer(int index) {

		return buffers[index];

	}


};