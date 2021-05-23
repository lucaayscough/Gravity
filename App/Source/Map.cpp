#include "Map.h"


//--------------------------------------------------//
// Constructors and destructors.

Map::Map(){
    // Planet destruction.
    m_DestroyPlanet.setValue(false);
    m_DestroyPlanet.addListener(this);
}

Map::~Map(){}


//--------------------------------------------------//
// Public methods.

void Map::paint(Graphics& g){
    g.fillAll(juce::Colours::black);
}

void Map::resized(){}


//--------------------------------------------------//
// Private methods.

void Map::createPlanet(int x, int y){
    Logger::writeToLog("Creating planet...");

    // Instantiate planet inside planets array.
    m_Planets.add(new Planet(&m_DestroyPlanet));
    
    m_NumPlanets = m_Planets.size();

    // Reference for ease of use.
    auto new_planet = m_Planets[m_NumPlanets - 1];

    // Render planet to screen.
    new_planet->setMapBoundaries(getWidth(), getHeight());
    
    new_planet->setBounds(
        x - (new_planet->getDiameter() / 2),
        y - (new_planet->getDiameter() / 2),
        new_planet->getDiameter(),
        new_planet->getDiameter()
    );
    
    addAndMakeVisible(new_planet);

    Logger::writeToLog("Planet created.");
    Logger::writeToLog("Number of planets: " + std::to_string(m_NumPlanets) + "\n");
}

void Map::destroyPlanet(){
    // For each planet check to see if it has been set for destruction.
    for(int i = 0; i < m_Planets.size(); i++){
        if(m_Planets[i]->m_Destroy == true){
            // Remove planet from array and delete.
            m_Planets.remove(i, true);

            // Change listener back to being false.
            m_DestroyPlanet.setValue(false);
        }
    }

    Logger::writeToLog("Planet destroyed.");
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

void Map::valueChanged(juce::Value &value){
    if(m_DestroyPlanet == true)
        destroyPlanet();
}