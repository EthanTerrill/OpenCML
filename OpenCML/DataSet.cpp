class dataSet {


private:
	std::vector<image> input;
	std::vector<image> output;
	std::vector<std::vector<unsigned int>> batch;
	unsigned int batchSize;


	unsigned int groupNum;
	unsigned int FileNum;


	void readFileError() {


		std::cerr << "\n\a";
		std::cerr << " _________________________"  << std::endl;
		std::cerr << "|        Error:           |" << std::endl;
		std::cerr << "|                         |" << std::endl;
		std::cerr << "|    Invalid.dtst File    |" << std::endl;
		std::cerr << " _________________________"  << std::endl;
		exit(0);

	}



public:

	
	 
	dataSet() {

		batchSize = 5;

	}

	/*
		takes as input a string with the path to a .dtst
	*/

	void load(std::string inputF, std::string outputF) {

		srand(time(NULL));
		std::vector<std::string> inputStr;
		std::vector<std::string> outputStr;

		for (const auto& entry : std::filesystem::directory_iterator(inputF)) {
			std::wstring wide = entry.path();
			inputStr.push_back(std::string(wide.begin(), wide.end()));

		}

		for (const auto& entry : std::filesystem::directory_iterator(outputF)) {
			std::wstring wide = entry.path();
			outputStr.push_back(std::string(wide.begin(), wide.end()));

		}


		if (inputStr.size() == outputStr.size()) {
			for (int i = 0; i < inputStr.size(); i++) {


				std::cout << inputStr[i] << std::endl;

				image inp;
				inp.open(inputStr[i]);
				input.push_back(inp);

				image outp;
				outp.open(outputStr[i]);
				output.push_back(outp);



			}

			int n = -1;
			for (int i = 0; i < inputStr.size(); i++) {

				
				if (i % batchSize == 0) {
					
					n++;

					batch.push_back(std::vector<unsigned int>());
				
				}

				

				bool alreadyPicked = true;
				int a;
				while (alreadyPicked) {
				
					alreadyPicked = false;
					a = rand() % input.size();

					for (int j = 0; j < batch.size(); j++)
						for (int k = 0; k < batch[j].size(); k++)
							alreadyPicked = alreadyPicked || (a == batch[j][k]);


				}
				batch[n].push_back(a);



			}


		}
		else{
		
			std::cerr << "bad Data, Incompatible datapoints";
		}



	}


	void loadDataSetFile(std::string datasetF) {}

	image getInput(unsigned int i) {
		
		if (i < input.size() ) {
			return input[i];
		}
		else {
			//add error code 
		}
	}

	image getOuput(unsigned int i) {

		if (i < input.size()) {
			return output[i];
		}
		else {
			//add error code 
		}
	}



	void trainOnBatch(CNN network, int n) {
		
		for (int i = 0; i < batch[n].size(); i++) {
			network.forwardPropagate(input[batch[n][i]]);
			network.backpropagate(input[batch[n][i]], output[batch[n][i]]);
					
		}
		network.update();
		
	}
	int getBatchNum() {
		return batch.size();
	}

};