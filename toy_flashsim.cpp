#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>


#include <torch/script.h>

using namespace std;

// Function to linearize a 2D vector
vector<float> linearize(const vector<vector<float>>& vec_vec)
{
	vector<float> vec;
	for (const auto& v : vec_vec) {
		for (auto d : v) {
			vec.push_back(d);
		}
	}
	return vec;
}

int main() {

	int row_size_context0;
	int col_size_context0;
	
	string fname_context0 = "../context0.csv"; // Change this with the directory where the CSV files are located
	vector<vector<string>> context0_CSV;
	vector<string> row_context0;
	string line_context0, word_context0;
 
	fstream file_context0 (fname_context0, ios::in);
	if(file_context0.is_open())   {
		while(getline(file_context0, line_context0))
		{
			row_context0.clear();
 
			stringstream str(line_context0);
 
			while(getline(str, word_context0, ','))
				row_context0.push_back(word_context0);
			context0_CSV.push_back(row_context0);
		}
	}
	else
		cout<<"Could not open the context file\n";
	
	row_size_context0 = context0_CSV.size();
 
	for(int i=0;i<context0_CSV.size();i++)   {
		col_size_context0 = context0_CSV[i].size();
	}
	
	vector<vector<float>> context0(row_size_context0, vector<float>(col_size_context0));
	
	for(int i=0;i<context0_CSV.size();i++)   {
		for(int j=0;j<context0_CSV[i].size();j++)
		{
			context0[i][j] = stof(context0_CSV[i][j]);
			cout<< context0[i][j]<<" ";
		}
		cout<<"\n";
	}
	

	
	// Load pretrained flow model
	auto flow = torch::jit::load("../flow_model.pt"); // Change this with the directory where your model is located
	flow.eval();
	
	//Make context0 readable by torch
	
	vector<float> vec_context0 = linearize(context0);
	torch::Tensor t_context0 = torch::from_blob(vec_context0.data(), {row_size_context0,col_size_context0});
	cout << "\n t_context = \n" << t_context0 << std::endl;
	
	int num_iter = 200;
	
	for (int i = 0; i < num_iter; ++i) {
		// Do some interesting stuff here
		if ((i + 1) % 100 == 0) {
			// Sample from prior
			auto samples0 = flow({t_context0}).toTensor().exp();
			cout << "\n samples = \n" << samples0 << std::endl;
			
			// Get the size of the tensor
			auto size0 = samples0.sizes();
			
			// Save size-x and size-z to int variables
			int size0_x = size0[0]; 
			int size0_z = size0[2];
			
			cout << "\n size_x = " << size0_x << "\n size_z = " << size0_z << endl;
			
			//Save samples to a CSV file
			ofstream file_samples0("samples0_Cpp.csv");
			
			for (int i = 0; i < size0_x; ++i)  {
				for (int j = 0; j < size0_z; ++j)  {
					file_samples0 << samples0[i][0][j].cpu().item<float>() << ",";
				}
				file_samples0 << "\n";
			}

			file_samples0.close();
			
			
			//Read samples0_Cpp.csv and copy it to a vector
			int row_size_samples0;
			int col_size_samples0;
			
			string fname_samples0 = "samples0_Cpp.csv";
			vector<vector<string>> samples0_CSV;
			vector<string> row_samples0;
			string line_samples0, word_samples0;
			
			fstream file_samples0_read (fname_samples0, ios::in);
			if(file_samples0_read.is_open())   {
				while(getline(file_samples0_read, line_samples0))   {
					row_samples0.clear();
					
					stringstream str(line_samples0);
					
					while(getline(str, word_samples0, ','))
						row_samples0.push_back(word_samples0);
					samples0_CSV.push_back(row_samples0);
				}
			}
			else
				cout<<"Could not open the samples0 file\n";
			
			row_size_samples0 = samples0_CSV.size();
			
			for(int i=0;i<samples0_CSV.size();i++)   {
				col_size_samples0 = samples0_CSV[i].size();
			}
			
			vector<vector<float>> samples0_vector(row_size_samples0, vector<float>(col_size_samples0));
			
			cout<<"\n samples0 = \n";
			
			for(int i=0;i<samples0_CSV.size();i++)   {
				for(int j=0;j<samples0_CSV[i].size();j++)   {
					samples0_vector[i][j] = stof(samples0_CSV[i][j]);
					cout<< samples0_vector[i][j]<<" ";
				}
				cout<<"\n";
			}

		}
	}
	
	
	return 0;
}
