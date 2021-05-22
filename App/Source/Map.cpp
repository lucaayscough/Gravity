#include "Map.h"


Map::Map(){}

Map::~Map(){}

void Map::paint(Graphics& g){
    g.fillAll(juce::Colours::black);
}

void Map::createPlanet(int x, int y){
    Logger::writeToLog("Creating planet...");
    m_Planets.emplace_back();
    Logger::writeToLog("Planet created.");
}

void Map::mouseDoubleClick(const MouseEvent& e){
    Logger::writeToLog("Detected double click.");

    int eventX = e.getMouseDownX();
    int eventY = e.getMouseDownY();

    createPlanet(eventX, eventY);
}
