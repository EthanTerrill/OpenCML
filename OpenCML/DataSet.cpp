
class dataSet {


private:
	std::vector<image> input;
	std::vector<image> output;


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

	void loadInputFolder(std::string inputF) {}

	void loadoutputFolder(std::string outputF) {}


	void loadDataSetFile(std::string datasetF) {}

};