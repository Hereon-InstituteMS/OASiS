#include <deal.II/base/config.h>
#include <deal.II/lac/vector.h>
#include <iostream>
int main(){
  std::cout << "dealii_version=" << DEAL_II_PACKAGE_VERSION << std::endl;
#ifdef DEBUG
  std::cout << "consumer_DEBUG=1" << std::endl;
#else
  std::cout << "consumer_DEBUG=0" << std::endl;
#endif
  dealii::Vector<double> v(3);
  std::cout << "before_invalid_index" << std::endl;
  std::cout << "v7=" << v[7] << std::endl;
  std::cout << "after_invalid_index" << std::endl;
}
