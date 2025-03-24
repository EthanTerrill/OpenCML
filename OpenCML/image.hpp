
#include "buffer.cpp"

class image {

private:
	unsigned int dimensionNum, width, height;


public:
	buffer* buffers;


	image();
	image(unsigned int width, unsigned int height);
	image(unsigned int width, unsigned int height, unsigned int dimensionNum);


	void open(std::string fileName);

	void toBmp(std::string fileName);

	int getDimensionNum();

	int getWidth();
	int getHeight();

	void wipe();
	void clear();
	void threshold(float threshold);
	buffer getBuffer(int index);
};
