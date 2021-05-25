#include "Headers.h"


// Main Planet class.

//--------------------------------------------------//
// Constructors and destructors.

Planet::Planet(){}

Planet::Planet(Generator* generator_ptr)
    : m_GeneratorPtr(generator_ptr)
    {
        // Allocates storage to array that holds sample.
        m_Sample.ensureStorageAllocated(m_GeneratorPtr->M_NUM_SAMPLES);

        generateLatents();
        generateSample();

        m_Destroy.setValue(false);
        //m_Destroy.addListener(this);
    }

Planet::~Planet(){}


//--------------------------------------------------//
// Public methods.

void Planet::paint(Graphics& g){
    g.setColour(juce::Colours::red);
    g.fillEllipse(0, 0, m_Diameter, m_Diameter);
}

void Planet::resized(){
    setSize(m_Diameter, m_Diameter);
}

void Planet::draw(int diameter, int x, int y){
    // When called the component is redrawn.

    setDiameter(diameter);
    setBounds(x, y, diameter, diameter);
}

void Planet::resizePlanet(int diameter){
    // When called the planet will be resized.
    // X and Y are calculated such that the planet will remain centred.

    int new_x;
    int new_y;

    if(diameter > getDiameter()){
        new_x = getX() - (M_SIZE_MODIFIER / 2);
        new_y = getY() - (M_SIZE_MODIFIER / 2);
    } else if(diameter < getDiameter()){
        new_x = getX() + (M_SIZE_MODIFIER / 2);
        new_y = getY() + (M_SIZE_MODIFIER / 2);
    }

    draw(diameter, new_x, new_y);
}

void Planet::setDiameter(int diameter){
    m_Diameter = diameter;
}

void Planet::setMapBoundaries(int width, int height){
    m_WindowWidth = width;
    m_WindowHeight = height;
}

int Planet::getDiameter(){
    return m_Diameter;
}

void Planet::generateLatents(){
    m_Latents = m_GeneratorPtr->generateLatents();
}

void Planet::generateSample(){
    m_Sample = m_GeneratorPtr->generateSample(m_Latents);
}


//--------------------------------------------------//
// Private methods.

void Planet::mouseDown(const MouseEvent& e){
    // Starts dragging component.
    m_Dragger.startDraggingComponent(this, e);
    
    if(e.getNumberOfClicks() > 1 && e.mods.isLeftButtonDown()){
        Logger::writeToLog("Generating sample...");

        generateLatents();
        generateSample();

        Logger::writeToLog("Sample generated.");
    }

    else if(e.mods.isLeftButtonDown() && e.mouseWasClicked()){
        Logger::writeToLog("Playing audio...");
        AudioContainer::audio.clear();
        AudioContainer::audio.addArray(m_Sample);
        AudioContainer::playAudio = true;
    }
    
    else if(e.mods.isRightButtonDown()){
        // Initializes planet destruction.
        m_Destroy.setValue(true);
        Logger::writeToLog("Set to destroy.");
    }    
}

void Planet::mouseDrag(const MouseEvent& e){
    if(e.mods.isLeftButtonDown()){
        m_Dragger.dragComponent(this, e, nullptr);
        checkBounds();
    }
}

void Planet::mouseWheelMove(const MouseEvent& e, const MouseWheelDetails& w){
    // If the mouse wheel is moved the diameter of the planet is
    // modified making sure it is not going over the size limitations.
    
    Logger::writeToLog("Wheel moved.");

    if(w.deltaY > 0.0f && getDiameter() < M_MAX_PLANET_SIZE)
        resizePlanet(getDiameter() + M_SIZE_MODIFIER);
    else if(w.deltaY < 0.0f && getDiameter() > M_MIN_PLANET_SIZE)
        resizePlanet(getDiameter() - M_SIZE_MODIFIER);
}

void Planet::visibilityChanged(){
    Logger::writeToLog("Visibility changed.");
}

void Planet::checkBounds(){
    auto posX = getX();
    auto posY = getY();

    if(posX < 0)
        setBounds(0, posY, m_Diameter, m_Diameter);

    posX = getX();
    posY = getY();

    if(posY < 0)
        setBounds(posX, 0, m_Diameter, m_Diameter);
    
    posX = getX();
    posY = getY();

    if(posX + m_Diameter > m_WindowWidth)
        setBounds(m_WindowWidth - m_Diameter, posY, m_Diameter, m_Diameter);

    posX = getX();
    posY = getY();

    if(posY + m_Diameter > m_WindowHeight)
        setBounds(posX, m_WindowHeight - m_Diameter, m_Diameter, m_Diameter);
    
    Logger::writeToLog("X: " + std::to_string(posX) + ", Y: " + std::to_string(posY));
}