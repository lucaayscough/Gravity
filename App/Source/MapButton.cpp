#include "Headers.h"


//------------------------------------------------------------//
// Constructors and destructors.

MapButton::MapButton(){
    Logger::writeToLog("Created MapButton.");
}

MapButton::~MapButton(){
    Logger::writeToLog("Destroyed MapButton.");
}

//------------------------------------------------------------//
// View methods.

void MapButton::paint(Graphics& g){
    g.fillAll(juce::Colours::white);
}

void MapButton::resized(){}
