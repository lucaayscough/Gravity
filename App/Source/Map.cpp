#include "Headers.h"


//--------------------------------------------------//
// Constructors and destructors.

Map::Map(AudioContainer& audiocontainer_ref, Parameters& parameters_ref)
    :   m_AudioContainerRef(audiocontainer_ref), m_ParametersRef(parameters_ref), m_ControlPanel(m_ParametersRef),
        m_Sun(m_AudioContainerRef, m_ParametersRef, m_ControlPanel){
    Logger::writeToLog("Map created.");

    addChildComponent(m_ControlPanel, -1);
    addChildAndSetID(&m_Sun, m_ParametersRef.SUN_ID);

    // TODO:
    // Clean this up.

    m_ForceVectorGradient.addColour((double)0.0, juce::Colours::darkred);
    m_ForceVectorGradient.addColour((double)0.2, juce::Colours::red);
    m_ForceVectorGradient.addColour((double)0.4, juce::Colours::orange);
    m_ForceVectorGradient.addColour((double)0.7, juce::Colours::yellow);
    m_ForceVectorGradient.addColour((double)1.0, juce::Colours::white);

    addListeners();
}

Map::~Map(){
    removeListeners();

    Logger::writeToLog("Map destroyed.");
}

//--------------------------------------------------//
// View methods.

void Map::paint(Graphics& g){
    int rect_overlap = 25;

    g.setGradientFill(m_BackgroundGradient);
    g.fillRect(0, 0, getWidth(), getHeight() / 2);
    g.fillRoundedRectangle(0, getHeight() / 2 - rect_overlap, getWidth(), getHeight() / 2 + rect_overlap, 5.0f);

    paintOrbits(g);
    paintForceVectors(g);
}

void Map::paintOrbits(Graphics& g){
    juce::ValueTree rootPlanetNode =  m_ParametersRef.getRootPlanetNode();
    juce::ValueTree sunNode =  m_ParametersRef.getSunNode();

    for(int i = 0; i < rootPlanetNode.getNumChildren(); i++){
        g.setColour(Variables::MAP_CIRCLE_COLOUR);
        g.drawEllipse(
            (getWidth() / 2) - (m_ParametersRef.getDistance(rootPlanetNode.getChild(i), sunNode)), 
            (getHeight() / 2) - (m_ParametersRef.getDistance(rootPlanetNode.getChild(i), sunNode) ),
            m_ParametersRef.getDistance(rootPlanetNode.getChild(i), sunNode) * 2,
            m_ParametersRef.getDistance(rootPlanetNode.getChild(i), sunNode) * 2,
            1
        );
    }
}

void Map::paintForceVectors(Graphics& g){
    // TODO:
    // Clean this up.


    g.setColour(juce::Colours::white);

    // Draw planet vectors.
    for(int i = 0; i < m_Planets.size(); i++){
        if(m_Planets[i]->m_ShowForceVectors.getValue() == juce::var(true)){
            auto planet_a = m_ParametersRef.getRootPlanetNode().getChildWithProperty(Parameters::idProp, m_Planets[i]->getComponentID());

            for(int j = 0; j < m_Planets.size(); j++){
                if(m_Planets[i]->getComponentID() == m_Planets[j]->getComponentID()){
                    continue;
                }
                auto planet_b = m_ParametersRef.getRootPlanetNode().getChildWithProperty(Parameters::idProp, m_Planets[j]->getComponentID());
                auto force_vector = m_ParametersRef.getForceVector(planet_a, planet_b);
                
                g.setOpacity(force_vector);
                g.drawLine(m_Planets[i]->getCentreX(), m_Planets[i]->getCentreY(), m_Planets[j]->getCentreX(), m_Planets[j]->getCentreY(), Variables::FORCE_VECTOR_SIZE);
            }

            auto force_vector = m_ParametersRef.getForceVector(m_ParametersRef.getSunNode(), planet_a);
            
            g.setOpacity(force_vector);
            g.drawLine(m_Planets[i]->getCentreX(), m_Planets[i]->getCentreY(), m_Sun.getCentreX(), m_Sun.getCentreY(), Variables::FORCE_VECTOR_SIZE);
        }
    }

    // Draw sun vectors.
    if(m_Sun.m_ShowForceVectors.getValue() == juce::var(true)){
        for(int i = 0; i < m_Planets.size(); i++){
            auto planet = m_ParametersRef.getRootPlanetNode().getChildWithProperty(Parameters::idProp, m_Planets[i]->getComponentID());
            auto force_vector = m_ParametersRef.getForceVector(m_ParametersRef.getSunNode(), planet);

            g.setOpacity(force_vector);
            g.drawLine(m_Sun.getCentreX(), m_Sun.getCentreY(), m_Planets[i]->getCentreX(), m_Planets[i]->getCentreY(), Variables::FORCE_VECTOR_SIZE);
        }
    }
}

void Map::resized(){
    drawSun();
    if(getNumPlanets() > 0){rebuildPlanets();}
    m_ControlPanel.setBounds(getLocalBounds());
    m_BackgroundGradient = juce::ColourGradient(Variables::MAP_BG_COLOUR_1, getWidth() / 2, getHeight() / 2, Variables::MAP_BG_COLOUR_2, getWidth() / 4, getHeight() / 4, true);
}

void Map::drawSun(){
    m_Sun.draw();
}

void Map::createPlanet(int x, int y){
    // TODO:
    // Clean this up.


    // TODO:
    // Add check for other astri.

    // Check creation position.
    int default_radius = (int)sqrt(Variables::DEFAULT_PLANET_AREA / 3.1415f);

    if(x - default_radius < 0){x = x + abs(x - default_radius);}
    else if(x + default_radius > getWidth()){x = x - ((x + default_radius) - getWidth());}
    if(y - default_radius < 0){y = y + abs(y - default_radius);}
    else if(y + default_radius > getHeight()){y = y - ((y + default_radius) - getHeight());}

    // Create planet node.
    m_ParametersRef.addPlanetNode();
    juce::ValueTree node = m_ParametersRef.getRootPlanetNode().getChild(m_ParametersRef.getRootPlanetNode().getNumChildren() - 1);

    // Instantiate planet inside planets array.
    m_Planets.add(new Planet(m_Planets, m_AudioContainerRef, m_ParametersRef, m_ControlPanel));

    // Extra setup for planet object.
    setupPlanet(m_Planets[getNumPlanets() - 1], x, y, node);
}

void Map::setupPlanet(Planet* planet, int x, int y, juce::ValueTree node){
    // TODO:
    // Clean this up.

    // Visibility.
    addChildAndSetID(planet, node.getProperty(Parameters::idProp));

    planet->draw(
        planet->getDiameter(),
        x - (planet->getDiameter() / 2) - (planet->getClipBoundary() / 2),
        y - (planet->getDiameter() / 2) - (planet->getClipBoundary() / 2)
    );

    planet->m_ShowForceVectors.addListener(this);

    planet->setPosXY(planet->getX(), planet->getY());
    planet->updateGraph();
}

void Map::destroyPlanet(juce::String& id){
    // TODO:
    // Clean this up.

    for(int i = 0; i < m_Planets.size(); i++){
        if(m_Planets[i]->getComponentID() == id){
            m_Planets[i]->m_ShowForceVectors.removeListener(this);
            m_Planets.remove(i, true);
        }
    }
}

void Map::rebuildPlanets(){
    // TODO:
    // Clean this up.

    for(int i = 0; i < getNumPlanets(); i++){
        // Create planet node.
        juce::ValueTree node = m_ParametersRef.getRootPlanetNode().getChild(i);

        // Instantiate planet inside planets array.
        m_Planets.add(new Planet(m_Planets, m_AudioContainerRef, m_ParametersRef, m_ControlPanel));
        m_Planets[i]->setComponentID(node.getProperty(Parameters::idProp));
        addAndMakeVisible(m_Planets[i]);
        m_Planets[i]->draw();
        m_Planets[i]->m_ShowForceVectors.addListener(this);
    }
}

//--------------------------------------------------//
// Interface methods.

int Map::getMaxNumPlanets(){return Variables::MAX_NUM_PLANETS;}
int Map::getNumPlanets(){return m_ParametersRef.getRootPlanetNode().getNumChildren();}

//--------------------------------------------------//
// Controller methods.

void Map::mouseUp(const MouseEvent& e){juce::ignoreUnused(e);}

void Map::mouseDoubleClick(const MouseEvent& e){
    Logger::writeToLog("Detected double click.");

    if(e.mods.isLeftButtonDown()){
        int eventX = e.getMouseDownX();
        int eventY = e.getMouseDownY();

        if(getNumPlanets() < getMaxNumPlanets()){
            createPlanet(eventX, eventY);
        }
        else{
            Logger::writeToLog("Maximum number of planets reached.");
        }
    }
}

//--------------------------------------------------//
// Callback methods.

void Map::addListeners(){
    m_ParametersRef.rootNode.addListener(this);
    m_ParametersRef.updateMap.addListener(this);
    m_Sun.m_ShowForceVectors.addListener(this);
}

void Map::removeListeners(){
    m_ParametersRef.rootNode.removeListener(this);
    m_ParametersRef.updateMap.removeListener(this);
    m_Sun.m_ShowForceVectors.removeListener(this);
}

void Map::valueChanged(juce::Value& value){
    juce::ignoreUnused(value);
    
    // Check if map needs updating.
    if(m_ParametersRef.updateMap.getValue() == juce::var(true)){
        rebuildPlanets();
        m_ParametersRef.updateMap.setValue(false);
    }

    // Check if force vectors need to be painted.
    for(int i = 0; i < m_Planets.size(); i++){
        if(m_Planets[i]->m_ShowForceVectors.getValue() == juce::var(true)){repaint();}
    }
    if(m_Sun.m_ShowForceVectors.getValue() == juce::var(true)){repaint();}
}

void Map::valueTreePropertyChanged(juce::ValueTree& node, const juce::Identifier& id){
    juce::ignoreUnused(node);
    if(id == Parameters::isActiveProp){repaint();}
    if(id == Parameters::posXProp || id == Parameters::posYProp){repaint();}
}

void Map::valueTreeChildRemoved(juce::ValueTree& parentNode, juce::ValueTree& removedNode, int index){
    juce::ignoreUnused(parentNode, index);
    if(removedNode.getType() == Parameters::planetType){
        juce::String id = removedNode.getProperty(Parameters::idProp).toString();
        destroyPlanet(id);
        repaint();
    }
}
