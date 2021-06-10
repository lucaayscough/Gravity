#include "Headers.h"


//--------------------------------------------------//
// Constructors and destructors.

Map::Map(AudioContainer* audiocontainer_ptr, Parameters& parameters)
    :   m_AudioContainerPtr(audiocontainer_ptr),
        m_ParametersRef(parameters),
        m_Sun(m_Planets, m_AudioContainerPtr, m_ParametersRef){
    Logger::writeToLog("Map created!");

    m_ParametersRef.updateMap.addListener(this);
    if(getNumPlanets() > 0){rebuildPlanets();}
}

Map::~Map(){
    for(int i = 0; i < m_Planets.size(); i++){
        m_Planets[i]->m_Destroy.removeListener(this);
    }

    Logger::writeToLog("Map destroyed!");
}

//--------------------------------------------------//
// Public methods.

void Map::paint(Graphics& g){g.fillAll(juce::Colours::black);}
void Map::resized(){createSun();}

void Map::createSun(){
    addAndMakeVisible(m_Sun);
    m_Sun.draw();
    m_Sun.setPosXY(m_Sun.getX(), m_Sun.getY());
    m_Sun.setCentrePosXY(m_Sun.getCentreX(&m_Sun), m_Sun.getCentreY(&m_Sun));
}

//--------------------------------------------------//
// Private methods.

void Map::createPlanet(int x, int y){
    // Create planet node.
    m_ParametersRef.addPlanetNode();
    juce::ValueTree node = m_ParametersRef.getRootPlanetNode().getChild(m_ParametersRef.getRootPlanetNode().getNumChildren() - 1);

    // Instantiate planet inside planets array.
    m_Planets.add(new Planet(m_Planets, m_AudioContainerPtr, m_ParametersRef));

    // Extra setup for planet object.
    setupPlanet(m_Planets[getNumPlanets() - 1], x, y, node);
}

void Map::setupPlanet(Planet* planet, int x, int y, juce::ValueTree node){
    // ID
    planet->setComponentID(node.getProperty(Parameters::idProp));

    // Visibility.
    planet->setMapSize(getWidth(), getHeight());
    addAndMakeVisible(planet);

    planet->draw(
        planet->getDiameter(),
        x - (planet->getDiameter() / 2) - (planet->getClipBoundary() / 2),
        y - (planet->getDiameter() / 2) - (planet->getClipBoundary() / 2)
    );

    planet->setPosXY(planet->getX(), planet->getY());
    planet->setCentrePosXY(planet->getCentreX(planet), planet->getCentreY(planet));

    planet->m_Destroy.addListener(this);
    planet->updateGraph();
}

void Map::destroyPlanet(){
    // For each planet check to see if it has been set for destruction.
    for(int i = 0; i < m_Planets.size(); i++){
        if(m_Planets[i]->m_Destroy == true){
            m_ParametersRef.removePlanetNode(m_Planets[i]->getComponentID());
            m_Planets.remove(i, true);
        }
    }
}

void Map::rebuildPlanets(){
    for(int i = 0; i < getNumPlanets(); i++){
        // Create planet node.
        juce::ValueTree node = m_ParametersRef.getRootPlanetNode().getChild(i);

        // Instantiate planet inside planets array.
        m_Planets.add(new Planet(m_Planets, m_AudioContainerPtr, m_ParametersRef));
        
        m_Planets[i]->setComponentID(node.getProperty(Parameters::idProp));

        addAndMakeVisible(m_Planets[i]);

        // Add listener for planet destruction request and lerp graph.
        m_Planets[i]->m_Destroy.addListener(this);
        
        m_Planets[i]->draw();
    }
}

int Map::getMaxNumPlanets(){return Variables::MAX_NUM_PLANETS;}
int Map::getNumPlanets(){return m_ParametersRef.getRootPlanetNode().getNumChildren();}

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

void Map::mouseUp(const MouseEvent& e){juce::ignoreUnused(e);}

void Map::mouseDoubleClick(const MouseEvent& e){
    Logger::writeToLog("Detected double click.");

    if(e.mods.isLeftButtonDown()){
        int eventX = e.getMouseDownX();
        int eventY = e.getMouseDownY();

        if(getNumPlanets() < getMaxNumPlanets())
            createPlanet(eventX, eventY);
        else
            Logger::writeToLog("Maximum number of planets reached.");
    }
}

void Map::valueChanged(juce::Value& value){
    juce::ignoreUnused(value);
    destroyPlanet();
    if(m_ParametersRef.updateMap.getValue() == juce::var(true)){
        rebuildPlanets();
        m_ParametersRef.updateMap.setValue(false);
    }
}
