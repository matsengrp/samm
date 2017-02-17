#ifndef OPTION_HPP
#define OPTION_HPP

#include <memory>
#include <vector>
using namespace std;

template <typename T>
class Option {
public:
	Option(): has_value{false} {};
	Option(vector<T> other): value{other}, has_value{true} {};
	bool is_set() {
    return has_value;
  };
	vector<T> get() {
    return value;
  };

private:
  bool has_value;
  vector<T> value;
};

#endif
