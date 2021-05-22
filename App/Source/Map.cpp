#include "Map.h"


// Constructors and destructors.

Map::Map(){
    reservePlanetMemory();
}

Map::~Map(){}


// Public methods.

void Map::paint(Graphics& g){
    g.fillAll(juce::Colours::black);
}

void Map::resized(){
    m_Planets.reserve(20);
}


// Private methods.

void Map::createPlanet(int x, int y){
    Logger::writeToLog("Creating planet...");

    // Instantiate planet inside planets vector.
    m_Planets.emplace_back();
    
    m_NumPlanets = m_Planets.size();

    // Reference for ease of use.
    Planet& planet = m_Planets.at(m_NumPlanets - 1);

    // Render planet to screen.
    planet.setMapBoundaries(getWidth(), getHeight());
    planet.setBounds(
        x - (planet.getDiameter() / 2),
        y - (planet.getDiameter() / 2),
        planet.getDiameter(),
        planet.getDiameter()
    );
    
    addAndMakeVisible(planet);

    Logger::writeToLog("Planet created.");
    Logger::writeToLog("Number of planets: " + std::to_string(m_NumPlanets) + "\n");
}

void Map::reservePlanetMemory(){
    // Sets the amount of memory to be reserved in vector for planet objects.
    m_Planets.reserve(M_MAX_NUM_PLANETS);
}

void Map::mouseDoubleClick(const MouseEvent& e){
    Logger::writeToLog("Detected double click.");

    int eventX = e.getMouseDownX();
    int eventY = e.getMouseDownY();

    if(m_NumPlanets < M_MAX_NUM_PLANETS)
        createPlanet(eventX, eventY);
    else
        Logger::writeToLog("Maximum number of planets reached.");
}
