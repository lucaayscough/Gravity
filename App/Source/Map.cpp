#include "Headers.h"


//--------------------------------------------------//
// Constructors and destructors.

Map::Map(AudioContainer& audiocontainer_ref, Parameters& parameters_ref)
    :   m_AudioContainerRef(audiocontainer_ref),
        m_ParametersRef(parameters_ref),
        m_ControlPanel(m_ParametersRef),
        m_Sun(m_Planets, m_AudioContainerRef, m_ParametersRef, m_ControlPanel){
    Logger::writeToLog("Map created!");

    addChildComponent(m_ControlPanel, -1);

    m_ParametersRef.rootNode.addListener(this);
    m_ParametersRef.updateMap.addListener(this);

    if(getNumPlanets() > 0){rebuildPlanets();}
}

Map::~Map(){
    m_ParametersRef.rootNode.removeListener(this);
    m_ParametersRef.updateMap.removeListener(this);

    Logger::writeToLog("Map destroyed!");
}

//--------------------------------------------------//
// View methods.

void Map::paint(Graphics& g){g.fillAll(juce::Colours::black);}
void Map::resized(){
    createSun();
    m_ControlPanel.setBounds(getLocalBounds());
}

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
    m_Planets.add(new Planet(m_Planets, m_AudioContainerRef, m_ParametersRef, m_ControlPanel));

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
    planet->updateGraph();
}

void Map::destroyPlanet(juce::String& id){
    for(int i = 0; i < getNumPlanets(); i++){
        if(m_Planets[i]->getComponentID() == id){
            m_Planets.remove(i, true);
        }
    }

    repaint();
}

void Map::rebuildPlanets(){
    for(int i = 0; i < getNumPlanets(); i++){
        // Create planet node.
        juce::ValueTree node = m_ParametersRef.getRootPlanetNode().getChild(i);

        // Instantiate planet inside planets array.
        m_Planets.add(new Planet(m_Planets, m_AudioContainerRef, m_ParametersRef, m_ControlPanel));
        
        m_Planets[i]->setComponentID(node.getProperty(Parameters::idProp));

        addAndMakeVisible(m_Planets[i]);
        
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

//--------------------------------------------------//
// Callback methods.

void Map::valueChanged(juce::Value& value){
    juce::ignoreUnused(value);
    
    if(m_ParametersRef.updateMap.getValue() == juce::var(true)){
        rebuildPlanets();
        m_ParametersRef.updateMap.setValue(false);
    }
}

void Map::valueTreePropertyChanged(juce::ValueTree& node, const juce::Identifier& id){
    juce::ignoreUnused(node);
    if(id == Parameters::isActiveProp){repaint();}
}

void Map::valueTreeChildRemoved(juce::ValueTree& parentNode, juce::ValueTree& removedNode, int index){
    juce::ignoreUnused(parentNode, index);
    if(removedNode.getType() == Parameters::planetType){
        juce::String id = removedNode.getProperty(Parameters::idProp).toString();
        destroyPlanet(id);
    }
}
