#include "Timer.h"


void Timer::reset()
{
	m_begin = clock_t::now();
}

double Timer::elapsed() const
{
	return std::chrono::duration_cast<second_t>(clock_t::now() - m_begin).count();
}

Timer::~Timer()
{
}
