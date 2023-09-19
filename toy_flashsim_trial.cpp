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

// Function to concatenate 2D arrays along axis = 0
vector<vector<float>> concatenateArrays(const vector<vector<float>>& arr1, const vector<vector<float>>& arr2)
{
	vector<vector<float>> result;
	
	// Add all elements from arr1 to result
	
	for (const auto& row : arr1) {
		result.push_back(row);
	}
	
	// Add all elements from arr2 to result
	
	for (const auto& row : arr2) {
		result.push_back(row);
	}
	
	return result;
}


int main() {
	
	int N;
	N = 6000;
	//Read context0.csv and copy it to a vector
	int row_size_context0;
	int col_size_context0;
	
	int row_size_final;
	
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
	
	//Read real.csv and copy it to a vector
	int row_size_real;
	int col_size_real;
	
	string fname_real = "../real.csv"; // Change this with the directory where the CSV files are located
	vector<vector<string>> real_CSV;
	vector<string> row_real;
	string line_real, word_real;
 
	fstream file_real (fname_real, ios::in);
	if(file_real.is_open())   {
		while(getline(file_real, line_real))
		{
			row_real.clear();
 
			stringstream str(line_real);
 
			while(getline(str, word_real, ','))
				row_real.push_back(word_real);
			real_CSV.push_back(row_real);
		}
	}
	else
		cout<<"Could not open the real file\n";
	
	row_size_real = real_CSV.size();
 
	for(int i=0;i<real_CSV.size();i++)   {
		col_size_real = real_CSV[i].size();
	}
	
	vector<vector<float>> real(row_size_real, vector<float>(col_size_real));
	cout<<"\n real = \n";
	
	for(int i=0;i<real_CSV.size();i++)   {
		for(int j=0;j<real_CSV[i].size();j++)
		{
			real[i][j] = stof(real_CSV[i][j]);
			cout<<real[i][j]<<" ";
		}
		cout<<"\n";
	}
	
	
	//Read YN.csv and copy it to a vector
	int row_size_YN;
	int col_size_YN;
	
	string fname_YN = "../YN.csv"; // Change this with the directory where the CSV files are located
	vector<vector<string>> YN_CSV;
	vector<string> row_YN;
	string line_YN, word_YN;
 
	fstream file_YN (fname_YN, ios::in);
	if(file_YN.is_open())   {
		while(getline(file_YN, line_YN))
		{
			row_YN.clear();
 
			stringstream str(line_YN);
 
			while(getline(str, word_YN, ','))
				row_YN.push_back(word_YN);
			YN_CSV.push_back(row_YN);
		}
	}
	else
		cout<<"Could not open the YN file\n";
	
	row_size_YN = YN_CSV.size();
 
	for(int i=0;i<YN_CSV.size();i++)   {
		col_size_YN = YN_CSV[i].size();
	}
	
	vector<vector<float>> YN(row_size_YN, vector<float>(col_size_YN));
	cout<< "\n Y[:N,:] = \n";
	
	for(int i=0;i<YN_CSV.size();i++)   {
		for(int j=0;j<YN_CSV[i].size();j++)
		{
			YN[i][j] = stof(YN_CSV[i][j]);
			cout<< YN[i][j]<<" ";
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

			string CSV_directory = "../";
			
			vector<vector<float>> samples_vector;

			for (int ii = 1; ii < N / 500; ii++) {
				string fname_context = CSV_directory + "context" + to_string(ii) + ".csv";
				vector<vector<string>> context_CSV;
				vector<string> row_context;
				string line_context, word_context;
				
				fstream file_context (fname_context, ios::in);
				
				if(file_context.is_open())   {
					while(getline(file_context, line_context))   {
						row_context.clear();
						stringstream str(line_context);
						
						while(getline(str, word_context, ','))   {
							row_context.push_back(word_context);
							context_CSV.push_back(row_context);
						}
					}
				}
				else
					cout<<"Could not open the context file\n";
				
				int row_size_context1;
				int col_size_context1;
				
				row_size_context1 = context_CSV.size();
				
				for(int i=0;i<context_CSV.size();i++)   {
					col_size_context1 = context_CSV[i].size();
				}
				
				vector<vector<float>> context(row_size_context1, vector<float>(col_size_context1));
				
				for(int i=0;i<context_CSV.size();i++)   {
					for(int j=0;j<context_CSV[i].size();j++)   {
						context[i][j] = stof(context_CSV[i][j]);
						cout<< context[i][j]<<" ";
					}
					cout<<"\n";
				}
					
				
				//Make context readable by torch
				vector<float> vec_context = linearize(context);
				torch::Tensor t_context = torch::from_blob(vec_context.data(), {row_size_context1,col_size_context1});
				cout << "\n t_context = \n" << t_context << std::endl;
				
				auto samples1 = flow({t_context}).toTensor().exp();
				cout << "\n samples = \n" << samples1 << std::endl;
				
				// Get the size of the tensor
				auto size1 = samples1.sizes();
				
				// Save size-x and size-z to int variables
				int size1_x = size1[0]; 
				int size1_z = size1[2];
				
				cout << "\n size_x = " << size1_x << "\n size_z = " << size1_z << endl;
				
				//Save samples to a CSV file
				ofstream file_samples1("samples1_Cpp.csv");
				
				for (int i = 0; i < size1_x; ++i)  {
					for (int j = 0; j < size1_z; ++j)  {
						file_samples1 << samples1[i][0][j].cpu().item<float>() << ",";
					}
					file_samples1 << "\n";
				}
				
				file_samples1.close();
				
				//Read samples1_Cpp.csv and copy it to a vector
				int row_size_samples1;
				int col_size_samples1;
				
				string fname_samples1 = "samples1_Cpp.csv";
				vector<vector<string>> samples1_CSV;
				vector<string> row_samples1;
				string line_samples1, word_samples1;
				
				fstream file_samples1_read (fname_samples1, ios::in);
				if(file_samples1_read.is_open())   {
					while(getline(file_samples1_read, line_samples1))   {
						row_samples1.clear();
						stringstream str(line_samples1);
						
						while(getline(str, word_samples1, ','))
							row_samples1.push_back(word_samples1);
						
						samples1_CSV.push_back(row_samples1);
					}
				}
				
				else
					cout<<"Could not open the samples1 file\n";
				
				row_size_samples1 = samples1_CSV.size();
				row_size_final= row_size_samples1;
				
				for(int i=0;i<samples1_CSV.size();i++)   {
					col_size_samples1 = samples1_CSV[i].size();
				}
				
				vector<vector<float>> samples1_vector(row_size_samples1, vector<float>(col_size_samples1));
				
				cout<<"\n samples1 = \n";
				
				for(int i=0;i<samples1_CSV.size();i++)   {
					for(int j=0;j<samples1_CSV[i].size();j++)   {
						samples1_vector[i][j] = stof(samples1_CSV[i][j]);
						cout<< samples1_vector[i][j]<<" ";
					}
					cout<<"\n";
				}
				
				samples_vector = concatenateArrays(samples0_vector, samples1_vector);
				samples0_vector = samples_vector;
				
				ofstream out("samples_Cpp.csv");
				
				for (auto& row : samples_vector) {
					for (auto col : row)
						out << col <<',';
					out << '\n';
				}
				out.close();
			}
			
		}
	}
	
	/*ifstream file("samples_Cpp.csv");
	
	// Variables
	string line; 
	
	// Temporary file to store shifted lines
	ofstream temp_file("temp.csv");  
		
	// Read each line
	while(getline(file, line)) {
		if(file.gcount() > row_size_final) {  // Keep reading till we reach the last row_size_samples1 lines
			continue;
		}
			
		// Write the line to temporary file
		temp_file << line << endl; 
	}
		
	// Close files
	file.close();
	temp_file.close();
	
	// Remove original file
	remove("samples_Cpp.csv"); 
		
	// Rename temporary file
	rename("temp.csv", "samples_Cpp.csv"); */
	
	
	return 0;
}
