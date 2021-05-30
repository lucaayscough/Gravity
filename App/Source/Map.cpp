#include "Headers.h"


//--------------------------------------------------//
// Constructors and destructors.

Map::Map(){}

Map::Map(Generator* generator_ptr)
    : m_GeneratorPtr(generator_ptr){}

Map::~Map(){}


//--------------------------------------------------//
// Public methods.

void Map::paint(Graphics& g){
    g.fillAll(juce::Colours::black);
}

void Map::resized(){
    createSun();
}

void Map::createSun(){
    // Create sun object.
    m_Sun.add(new Sun(&m_Planets, m_GeneratorPtr));

    // Display sun.
    addAndMakeVisible(m_Sun[0]);

    // Draws sun to the center of the screen.
    m_Sun[0]->draw();
}


//--------------------------------------------------//
// Private methods.

void Map::createPlanet(int x, int y){
    Logger::writeToLog("Creating planet...");

    // Instantiate planet inside planets array.
    // Pointers to sun, planets and generator objects are passed.
    m_Planets.add(new Planet(&m_Planets, m_GeneratorPtr));
    
    m_NumPlanets = m_Planets.size();

    setupPlanet(m_Planets[m_NumPlanets - 1], x, y);

    Logger::writeToLog("Planet created.");
    Logger::writeToLog("Number of planets: " + std::to_string(m_NumPlanets) + "\n");
}

void Map::setPlanetID(Planet* planet){
    // Generate random ID for component.
    auto randomID = juce::String(juce::Random::getSystemRandom().nextInt(100001));    

    // Check if ID is unique.
    for(int i = 0; i < m_NumPlanets - 1; i++){
        if(planet->getComponentID() == randomID){
            while(planet->getComponentID() == randomID){
                randomID = juce::String(juce::Random::getSystemRandom().nextInt(100000)); 
            }
        }
    }
    
    // Set ID.
    planet->setComponentID(randomID);
}

void Map::setupPlanet(Planet* planet, int x, int y){
    setPlanetID(planet);

    addAndMakeVisible(planet);

    // Add listener for planet destruction request.
    planet->m_Destroy.addListener(this);

    // Render planet to screen.
    planet->setMapBoundaries(getWidth(), getHeight());
    
    planet->draw(
        planet->getDiameter(),
        x - (planet->getDiameter() / 2) - (planet->getClipBoundary() / 2),
        y - (planet->getDiameter() / 2) - (planet->getClipBoundary() / 2)
    );
    
}

void Map::destroyPlanet(){
    // For each planet check to see if it has been set for destruction.
    for(int i = 0; i < m_Planets.size(); i++){
        if(m_Planets[i]->m_Destroy == true){
            // Remove planet from array and delete.
            m_Planets.remove(i, true);

            // Reduce number of planets counter.
            m_NumPlanets -= 1;
        }
    }

    Logger::writeToLog("Planet destroyed.");
}

void Map::mouseDoubleClick(const MouseEvent& e){
    Logger::writeToLog("Detected double click.");

    if(e.mods.isLeftButtonDown()){
        int eventX = e.getMouseDownX();
        int eventY = e.getMouseDownY();

        if(m_NumPlanets < M_MAX_NUM_PLANETS)
            createPlanet(eventX, eventY);
        else
            Logger::writeToLog("Maximum number of planets reached.");
    }
}

void Map::valueChanged(juce::Value& value){
    destroyPlanet();
}
