#include "Headers.h"


//--------------------------------------------------//
// Constructors and destructors.

Map::Map(){}

Map::Map(AudioContainer* audiocontainer_ptr, Parameters* parameters_ptr):
    m_AudioContainerPtr(audiocontainer_ptr),
    m_ParametersPtr(parameters_ptr),
    m_Sun(&m_Planets, m_AudioContainerPtr, m_ParametersPtr->rootNode.getChild(0)){}

Map::~Map(){}


//--------------------------------------------------//
// Public methods.

void Map::paint(Graphics& g){g.fillAll(juce::Colours::black);}
void Map::resized(){createSun();}

void Map::createSun(){
    addAndMakeVisible(m_Sun);
    m_Sun.draw();
    m_Sun.setPosXY(m_Sun.getX(), m_Sun.getY());
    m_Sun.setCentrePosXY(m_Sun.getCentreX(&m_Sun), m_Sun.getCentreY(&m_Sun));
    m_Sun.addSample();
}


//--------------------------------------------------//
// Private methods.

void Map::createPlanet(int x, int y){
    // Create planet node.
    m_ParametersPtr->addPlanetNode();
    juce::ValueTree node = m_ParametersPtr->rootNode.getChild(m_ParametersPtr->rootNode.getNumChildren() - 1);

    // Instantiate planet inside planets array.
    m_Planets.add(new Planet(&m_Planets, m_AudioContainerPtr, node));
    
    // Update number of planets.
    m_NumPlanets = m_Planets.size();

    // Extra setup for planet object.
    setupPlanet(m_Planets[m_NumPlanets - 1], x, y, node);

    // Run latent mixture algorithm.
    mixLatents();
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

    planet->setComponentID(randomID);
}

void Map::setupPlanet(Planet* planet, int x, int y, juce::ValueTree node){
    setPlanetID(planet);
    node.setProperty(Parameters::idProp, planet->getComponentID(), nullptr);
    node.setProperty(Parameters::mapWidthProp, getWidth(), nullptr);
    node.setProperty(Parameters::mapHeightProp, getHeight(), nullptr);

    addAndMakeVisible(planet);

    // Add listener for planet destruction request and lerp graph.
    planet->m_Destroy.addListener(this);
    planet->m_LerpGraph.addListener(this);
    
    planet->draw(
        planet->getDiameter(),
        x - (planet->getDiameter() / 2) - (planet->getClipBoundary() / 2),
        y - (planet->getDiameter() / 2) - (planet->getClipBoundary() / 2)
    );

    planet->setPosXY(planet->getX(), planet->getY());
    planet->setCentrePosXY(planet->getCentreX(planet), planet->getCentreY(planet));
}

void Map::destroyPlanet(){
    // For each planet check to see if it has been set for destruction.
    for(int i = 0; i < m_Planets.size(); i++){
        if(m_Planets[i]->m_Destroy == true){
            m_ParametersPtr->removePlanetNode(m_Planets[i]->getComponentID());
            // Remove planet from array and delete.
            m_Planets.remove(i, true);

            // Reduce number of planets counter.
            m_NumPlanets -= 1;
        }
    }
}

int Map::getMaxNumPlanets(){return Variables::MAX_NUM_PLANETS;}

float Map::getDistance(Sun& sun, Planet* planet){
    int centrePlanetX = planet->getCentreX(planet);
    int centrePlanetY = planet->getCentreY(planet);
    int centreSunX = sun.getX() + sun.getDiameter() / 2;
    int centreSunY = sun.getY() + sun.getDiameter() / 2;

    float a = (float)pow(centreSunX - centrePlanetX, 2.0f);
    float b = (float)pow(centreSunY - centrePlanetY, 2.0f);
    return sqrt(a + b);
}

float Map::getDistance(Planet* planet_a, Planet* planet_b){  
    int centreXA = planet_a->getCentreX(planet_a);
    int centreYA = planet_a->getCentreY(planet_a);
    int centreXB = planet_b->getCentreX(planet_b);
    int centreYB = planet_b->getCentreY(planet_b);

    float a = (float)pow(centreXB - centreXA, 2.0f);
    float b = (float)pow(centreYB - centreYA, 2.0f);

    return sqrt(a + b);
}

float Map::getForceVector(Sun& sun, Planet* planet){
    float r = getDistance(sun, planet);
    float m = ((float)sun.getDiameter() * (float)planet->getDiameter());
    
    Logger::writeToLog("Width: " + std::to_string(getWidth()));
    Logger::writeToLog("Distance: " + std::to_string(r));
    
    return (m / pow(r, 2.0f));
}

float Map::getForceVector(Planet* planet_a, Planet* planet_b){
    float r = getDistance(planet_a, planet_b);
    float m = ((float)planet_a->getDiameter() * (float)planet_b->getDiameter());
    
    Logger::writeToLog("Width: " + std::to_string(getWidth()));
    Logger::writeToLog("Distance: " + std::to_string(r));
    
    return (m / pow(r, 2.0f));
}

void Map::mixLatents(){m_ParametersPtr->mixLatents();}

void Map::mouseUp(const MouseEvent& e){}

void Map::mouseDoubleClick(const MouseEvent& e){
    Logger::writeToLog("Detected double click.");

    if(e.mods.isLeftButtonDown()){
        int eventX = e.getMouseDownX();
        int eventY = e.getMouseDownY();

        if(m_NumPlanets < getMaxNumPlanets())
            createPlanet(eventX, eventY);
        else
            Logger::writeToLog("Maximum number of planets reached.");
    }
}

void Map::valueChanged(juce::Value& value){
    destroyPlanet();
    mixLatents();
}
