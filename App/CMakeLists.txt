cmake_minimum_required(VERSION 3.10)
project(Gravity VERSION 0.0.1)

add_definitions(-DDONT_SET_USING_JUCE_NAMESPACE)

add_subdirectory(Dependencies/JUCE)
add_subdirectory(Source)