#include <iostream>
#include <string>
#include <bits/stdc++.h>
#include <fstream>
#include <sstream>
#include <bitset>
#include <cmath>
#include <complex>
#include <time.h>
#include <cstdlib>
#include <random>
#include <chrono>

#define NUM_ITERS 10

std::vector<std::complex<float>> int_to_complex(std::vector<int> input){
	
	std::vector<std::complex<float>> output;
	for(size_t i = 0; i < input.size(); i++){
		std::complex<float> element(static_cast<float>(input[i]), 0.0f);
		output.push_back(element);
	}
	return output;
}

std::vector<float> complex_to_real(std::vector<std::complex<float>> input){
	
	std::vector<float> output;
	for(size_t i = 0; i < input.size(); i++){
		float element = real(input[i]);
		output.push_back(element);
	}
	return output;
}

std::vector<float> complex_to_imag(std::vector<std::complex<float>> input){
	
	std::vector<float> output;
	for(size_t i = 0; i < input.size(); i++){
		float element = imag(input[i]);
		output.push_back(element);
	}
	return output;
}

int compute_bit_rev(int num, int num_bits){
	
	int rev = 0;
	//compute the bit reverse of num
	for (int j = num_bits - 1; j >= 0; j--){
		rev |= (num & 1) << j;
		num >>= 1;
	}
	return rev;
}

std::vector<std::complex<float>> bit_rev_perm(std::vector<std::complex<float>> input){
	
	int N = static_cast<int>(input.size());
	int num_bits = static_cast<int>(log2(N));
	std::vector<std::complex<float>> output(N);
	int f_index;
	for (int s_index = 0; s_index < N; s_index++){
		f_index = compute_bit_rev(s_index, num_bits);
		output[f_index] = input[s_index];
	}
	return output;
}

std::vector<std::complex<float>> compute_fft(std::vector<std::complex<float>> input){
	
	int N = static_cast<int>(input.size());
	int pad = static_cast<int>(exp2(ceil(log2(N)))) - N;
	const std::complex<float> comp_zero(0.0f, 0.0f);
	for (int i = 0; i < pad; i++)
		input.push_back(comp_zero);
	N = static_cast<int>(input.size());
	std::vector<std::complex<float>> output = bit_rev_perm(input);
	const float pi = std::acos(-1);
    	const std::complex<float> comp_unit(0.0f, 1.0f);
    	const int log2_N = static_cast<int>(log2(N));
    	int m;
    	std::complex<float> w_m, w, u, t;
	for (int l = 1; l <= log2_N; l++){
		m = static_cast<int>(exp2(l));
		w_m = std::exp((-2*pi*comp_unit)/static_cast<float>(m));
		for (int i = 0; i <= N-1; i += m){
			w = std::complex<float>(1.0f, 0.0f);
			for (int j = 0; j <= (m/2) - 1; j++){
				u = output[i + j];
				t = w*output[i + j + (m/2)];
				output[i + j] = u + t;
				output[i + j + (m/2)] = u - t;
				w *= w_m ;
			}
		}
	}
	return output;
}

//we cannot access the columns of a vector of vectors directly. we can however access the 
//rows, each row representing a vector. thus, in order to access the columns of a vector, we 
//can transpose it, access the rows, and then transpose it back
template<typename T>
std::vector<std::vector<T>> vector_transpose(std::vector<std::vector<T>> input){
	
	std::vector<std::vector<T>> output;
	int num_rows = static_cast<int>(input.size());
	int num_cols = static_cast<int>(input[0].size());
	for(int col = 0; col < num_cols; col++){
		std::vector<T> input_col;
		for(int row = 0; row < num_rows; row++){
			input_col.push_back(input[row][col]);
		}
		output.push_back(input_col);
	}
	return output;
}

//first index -> row; second index -> column
std::vector<std::vector<std::complex<float>>> compute_2dfft(
	
	std::vector<std::vector<std::complex<float>>> input){
	for(int row_index = 0; row_index < static_cast<int>(input.size()); row_index++){
		input[row_index] = compute_fft(input[row_index]);
	}
	input = vector_transpose(input);
	for(int col_index = 0; col_index < static_cast<int>(input.size()); col_index++){
		input[col_index] = compute_fft(input[col_index]);
	}
	return vector_transpose(input);
}

//first index -> row; second index -> column
std::vector<std::vector<std::complex<float>>> compute_2difft(
	
	std::vector<std::vector<std::complex<float>>> input){
	int num_rows = static_cast<int>(input.size());
	int num_cols = static_cast<int>(input[0].size());
	//CONJUGATE INPUT
	for(int row_index = 0; row_index < num_rows; row_index++){
		for(int col_index = 0; col_index < num_cols; col_index++){
			input[row_index][col_index] = std::conj(input[row_index][col_index]);
		}
	}
	input = compute_2dfft(input);
	num_rows = static_cast<int>(input.size());
	num_cols = static_cast<int>(input[0].size());
	//CONJUGATE AND SCALE OUTPUT
	for(int row_index = 0; row_index < num_rows; row_index++){
		for(int col_index = 0; col_index < num_cols; col_index++){
			input[row_index][col_index] = 
			std::conj(input[row_index][col_index]) / static_cast<float>(num_rows*num_cols);
		}
	}
	return input;
}

//argv[1] should be the filename of the 2D signal whose 2d-dft should be computed, once the
//filename is given, a random signal will be generated and stored in said file;
//argv[2] should be the vertical resolution; 
//argv[3] should be the horizontal resolution; 
int main(int argc, char const *argv[])
{
	//GENERATE A RANDOM 2D SIGNAL WHOSE DFT WILL BE COMPUTED AND EXPORT IT
	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_int_distribution<int> uni(0, 255);
	const int num_rows = atoi(argv[2]);
	const int num_cols = atoi(argv[3]);
	const std::string input_filename = argv[1];
	std::ofstream output_input_file;
	output_input_file.open(input_filename);
	for(int row = 0; row < num_rows; row++){
		for(int col = 0; col < num_cols; col++){
			output_input_file << uni(rng) << " ";
		}
		output_input_file << std::endl;
	}
	output_input_file.close();
	std::cout << "Generated random signal and exported it to " << input_filename << "..." << std::endl;

	//READ THE INPUT FILE CONTAINING THE SIGNAL TO BE TRANSFORMED INTO THE FREQUENCY DOMAIN
	std::vector<std::vector<int>> input(num_rows, std::vector<int>(num_cols));
	std::ifstream input_file;
	input_file.open(input_filename);
	for(int row = 0; row < num_rows; row++){
		for(int col = 0; col < num_cols; col++){
			int pixel_val;
			input_file >> pixel_val;
			input[row][col] = pixel_val;
		}
	}
	input_file.close();
	std::cout << "Loaded input signal..." << std::endl;

	//TRANFORM INPUT INTO THE COMPLEX DOMAIN
	std::vector<std::vector<std::complex<float>>> input_complex;
	for(int row_index = 0; row_index < num_rows; row_index++){
		input_complex.push_back(int_to_complex(input[row_index]));
	}
	std::cout << "Converted signal into the complex domain..." << std::endl;

	//COMPUTE THE FFT AND MEASURE ITS EXECUTION TIME
	int avg_time = 0;
	std::vector<std::vector<std::complex<float>>> input_dft;
	for(int iter = 0; iter < NUM_ITERS; iter++){
		auto start_2dfft = std::chrono::high_resolution_clock::now();
		input_dft = compute_2dfft(input_complex);
		auto stop_2dfft = std::chrono::high_resolution_clock::now();
		auto duration_2dfft = std::chrono::duration_cast<std::chrono::microseconds>(stop_2dfft - start_2dfft);
		avg_time += static_cast<int>(duration_2dfft.count());
	}
	avg_time /= NUM_ITERS;
	std::cout << "Computed 2D DFT of the signal (avg. of " << NUM_ITERS << " iterations: " << avg_time << "us)..." << std::endl;

	//COMPUTE THE IFFT AND MEASURE ITS EXECUTION TIME
	avg_time = 0;
	std::vector<std::vector<std::complex<float>>> input_idft;
	for(int iter = 0; iter < NUM_ITERS; iter++){
		auto start_2difft = std::chrono::high_resolution_clock::now();
		input_idft = compute_2difft(input_dft);
		auto stop_2difft = std::chrono::high_resolution_clock::now();
		auto duration_2difft = std::chrono::duration_cast<std::chrono::microseconds>(stop_2difft - start_2difft);
		avg_time += static_cast<int>(duration_2difft.count());
	}
	avg_time /= NUM_ITERS;
	std::cout << "Computed 2D IDFT of the signal (avg. of " << NUM_ITERS << " iterations: " << avg_time << "us)..." << std::endl;

	//TRANFORM RESULT INTO THE REAL DOMAIN
	std::vector<std::vector<float>> output_real;
	for(int row_index = 0; row_index < static_cast<int>(input_idft.size()); row_index++){
		output_real.push_back(complex_to_real(input_idft[row_index]));
	}
	std::cout << "Converted signal into the real domain (ignored imaginary part)..." << std::endl;

	//EXPORT REAL RESULT
	std::ofstream real_output_file;
	std::string real_output_filename = "real_output_" + input_filename;
	real_output_file.open(real_output_filename);
	for(size_t row = 0; row < output_real.size(); row++){
		for(size_t col = 0; col < output_real[0].size(); col++){
			real_output_file << output_real[row][col] << " ";
		}
		real_output_file << std::endl;
	}
	real_output_file.close();
	std::cout << "Exported real result (" << output_real.size() << ", " << output_real[0].size() << ") to "
		<< real_output_filename << "..." << std::endl;

	//TRANFORM RESULT INTO THE IMAGINARY DOMAIN
	std::vector<std::vector<float>> output_imag;
	for(int row_index = 0; row_index < static_cast<int>(input_idft.size()); row_index++){
		output_imag.push_back(complex_to_imag(input_idft[row_index]));
	}
	std::cout << "Converted signal into the imaginary domain (ignored real part)..." << std::endl;
	
	//EXPORT RESULT
	std::ofstream imag_output_file;
	std::string imag_output_filename = std::string("imag_output_") + input_filename;
	imag_output_file.open(imag_output_filename);
	for(size_t row = 0; row < output_imag.size(); row++){
		for(size_t col = 0; col < output_imag[0].size(); col++){
			imag_output_file << output_imag[row][col] << " ";
		}
		imag_output_file << std::endl;
	}
	imag_output_file.close();
	std::cout << "Exported imaginary result (" << output_imag.size() << ", " << output_imag[0].size() << ") to "
		<< imag_output_filename << "..." << std::endl;

	return 0;
}
