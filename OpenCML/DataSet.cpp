class dataGroup {

private:

	image* input;
	image* output;

	unsigned int dataPairs;
	bool PairNumFound = false;

	void readFileError() {

		std::cerr << "\n\a";
		std::cerr << " _________________________" << std::endl;
		std::cerr << "|        Error:           |" << std::endl;
		std::cerr << "|                         |" << std::endl;
		std::cerr << "|   Invalid .dtst File    |" << std::endl;
		std::cerr << " _________________________" << std::endl;
		exit(0);

	}

public:


	void GetDSTfilePart(std::ifstream *file) {

		std::string k = " ";
		char c = 0;
		unsigned int imageId = 0;
		file[0] >> c;
		file[0] >> c;
		
		while (!file[0].eof() && k[k.size() - 1] != '}') {

			file[0] >> k;

			if (k[0] == '/' && k[k.size() - 1] != '\n') {
				while (c != '\n' && !file[0].eof()) {
					file[0].get(c);
				}
			}
			else {

				

				if ((k == "DataPairs:" || (k == "dataPairs:"))  && !PairNumFound) {

					file[0] >> dataPairs;

					input = new image[dataPairs];
					output = new image[dataPairs];


				}
				else if (k == "Pair:" || k == "pair:") 
				{

					file[0] >> k;

					if (k == "Input:" || k == "input:") {
						file[0] >> c;
						file[0] >> k;
						input[imageId] = image();
						input[imageId].open(k);

						k = k.substr(0,k.length() - 1);
						file[0] >> k;
						if (k == "output:" || k == "Output:") {
							
							file[0] >> c;
							file[0] >> k;
							output[imageId] = image();
							output[imageId].open(k);

							k = k.substr(0, k.length() - 1);
							file[0] >> k;
						}

					}
					else {
						readFileError();
					}
				}
			}
		}

	}
	dataGroup() {
	

	}



};

class dataSet {


private:
	dataGroup* groups;
	unsigned int groupNum;
	unsigned int FileNum;


	void readFileError() {
		

		std::cerr << "\n\a";
		std::cerr << " _________________________" << std::endl;
		std::cerr << "|        Error:           |" << std::endl;
		std::cerr << "|                         |" << std::endl;
		std::cerr << "|    Invalid.dtst File    |" << std::endl;
		std::cerr << " _________________________" << std::endl;
		exit(0);

	}

	

public:


	dataSet() {
	
	
	}

	/*
		takes as input a string with the path to a .dtst
	*/
	void open(std::string fileName) {
		
		std::ifstream f;
		f.open(fileName);

		char c = 0;
		unsigned int groupId = 0;
		while (!f.eof() && f.good() ) {
			
			f.get(c);
			if (c == '/') {
				while (c != '\n' && !f.eof()) {
					f.get(c);

				}
			}
			else {

				if (c == 'G' || c == 'g') {
					std::string k;
					f >> k;
					k = "G" + k;
					//std::cout << k;

					if (k == "GroupNum:") {
						
						f >> groupNum;

						groups = new dataGroup[groupNum];
					}
					else if (k == "Group") {
						
						groupId++;

						int check;

						f >> check;

						if (check == groupId) {
						
							//std::cout << check<<"\n";
							groups[groupId - 1].GetDSTfilePart(&f);


							
						}
						else {
						
							readFileError();
						
						}

						
					}


				}
				//std::cout << c;
			}
			
		
		
		}



		



	
	}



};