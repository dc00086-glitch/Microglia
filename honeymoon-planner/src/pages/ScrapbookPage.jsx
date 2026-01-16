import { useState } from 'react';
import { useHoneymoon } from '../context/HoneymoonContext';
import {
  BookHeart, Plus, Trash2, Edit2, MapPin, Calendar, Heart, Sparkles,
  Sun, Cloud, Camera, X, Save, Image
} from 'lucide-react';
import { format, parseISO } from 'date-fns';

const moods = [
  { value: 'romantic', label: 'Romantic', emoji: '💕', color: '#ff6b6b' },
  { value: 'adventurous', label: 'Adventurous', emoji: '🎒', color: '#4ecdc4' },
  { value: 'relaxed', label: 'Relaxed', emoji: '😌', color: '#95e1d3' },
  { value: 'excited', label: 'Excited', emoji: '🎉', color: '#ffd93d' },
  { value: 'grateful', label: 'Grateful', emoji: '🙏', color: '#a8e6cf' },
  { value: 'amazed', label: 'Amazed', emoji: '🤩', color: '#dda0dd' },
];

function MemoryCard({ memory, onEdit, onDelete }) {
  const mood = moods.find(m => m.value === memory.mood) || moods[0];

  return (
    <div className="memory-card" style={{ '--mood-color': mood.color }}>
      <div className="memory-header">
        <div className="memory-mood">
          <span className="mood-emoji">{mood.emoji}</span>
          <span className="mood-label">{mood.label}</span>
        </div>
        <div className="memory-actions">
          <button className="icon-btn" onClick={() => onEdit(memory)}>
            <Edit2 size={16} />
          </button>
          <button className="icon-btn delete" onClick={() => onDelete(memory.id)}>
            <Trash2 size={16} />
          </button>
        </div>
      </div>

      <h3 className="memory-title">{memory.title}</h3>

      <div className="memory-meta">
        <span className="memory-date">
          <Calendar size={14} />
          {format(parseISO(memory.date), 'MMMM d, yyyy')}
        </span>
        <span className="memory-location">
          <MapPin size={14} />
          {memory.city}
        </span>
      </div>

      <p className="memory-description">{memory.description}</p>

      {memory.photos && memory.photos.length > 0 && (
        <div className="memory-photos">
          {memory.photos.map((photo, index) => (
            <div key={index} className="photo-thumbnail">
              <img src={photo} alt={`Memory ${index + 1}`} />
            </div>
          ))}
        </div>
      )}

      <div className="memory-footer">
        <Heart size={14} fill={mood.color} color={mood.color} />
        <span>A moment to remember forever</span>
      </div>
    </div>
  );
}

export default function ScrapbookPage() {
  const { scrapbook, addMemory, updateMemory, deleteMemory, tripInfo } = useHoneymoon();
  const [showForm, setShowForm] = useState(false);
  const [editingMemory, setEditingMemory] = useState(null);
  const [formData, setFormData] = useState({
    date: '',
    city: '',
    title: '',
    description: '',
    mood: 'romantic',
    photos: [],
  });

  const resetForm = () => {
    setFormData({
      date: '',
      city: '',
      title: '',
      description: '',
      mood: 'romantic',
      photos: [],
    });
    setEditingMemory(null);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (editingMemory) {
      updateMemory(editingMemory.id, formData);
    } else {
      addMemory(formData);
    }
    resetForm();
    setShowForm(false);
  };

  const handleEdit = (memory) => {
    setFormData({
      date: memory.date,
      city: memory.city,
      title: memory.title,
      description: memory.description,
      mood: memory.mood,
      photos: memory.photos || [],
    });
    setEditingMemory(memory);
    setShowForm(true);
  };

  const handlePhotoUpload = (e) => {
    const files = Array.from(e.target.files);
    files.forEach(file => {
      const reader = new FileReader();
      reader.onloadend = () => {
        setFormData(prev => ({
          ...prev,
          photos: [...prev.photos, reader.result]
        }));
      };
      reader.readAsDataURL(file);
    });
  };

  const removePhoto = (index) => {
    setFormData(prev => ({
      ...prev,
      photos: prev.photos.filter((_, i) => i !== index)
    }));
  };

  const sortedMemories = [...scrapbook].sort((a, b) =>
    new Date(b.date) - new Date(a.date)
  );

  return (
    <div className="scrapbook-page">
      <header className="page-header scrapbook-header">
        <div className="header-content">
          <BookHeart size={32} />
          <div>
            <h1>Our Scrapbook</h1>
            <p>Capture and cherish your precious memories</p>
          </div>
        </div>
        <button className="btn-primary" onClick={() => { resetForm(); setShowForm(true); }}>
          <Plus size={18} /> Add Memory
        </button>
      </header>

      <div className="scrapbook-intro">
        <Sparkles size={24} />
        <p>Every adventure deserves to be remembered. Document your journey through Europe, one magical moment at a time.</p>
      </div>

      {showForm && (
        <div className="modal-overlay" onClick={() => { setShowForm(false); resetForm(); }}>
          <div className="modal memory-modal" onClick={(e) => e.stopPropagation()}>
            <h2>{editingMemory ? 'Edit Memory' : 'Create New Memory'}</h2>
            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label>Title</label>
                <input
                  type="text"
                  placeholder="Give this memory a title..."
                  value={formData.title}
                  onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                  required
                />
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label>Date</label>
                  <input
                    type="date"
                    value={formData.date}
                    onChange={(e) => setFormData({ ...formData, date: e.target.value })}
                    required
                  />
                </div>
                <div className="form-group">
                  <label>City</label>
                  <select
                    value={formData.city}
                    onChange={(e) => setFormData({ ...formData, city: e.target.value })}
                    required
                  >
                    <option value="">Select a city</option>
                    {tripInfo.destinations.map(city => (
                      <option key={city} value={city}>{city}</option>
                    ))}
                    <option value="Other">Other</option>
                  </select>
                </div>
              </div>

              <div className="form-group">
                <label>How did you feel?</label>
                <div className="mood-selector">
                  {moods.map(mood => (
                    <button
                      key={mood.value}
                      type="button"
                      className={`mood-btn ${formData.mood === mood.value ? 'active' : ''}`}
                      onClick={() => setFormData({ ...formData, mood: mood.value })}
                      style={{ '--btn-color': mood.color }}
                    >
                      <span className="mood-emoji">{mood.emoji}</span>
                      <span>{mood.label}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="form-group">
                <label>What happened?</label>
                <textarea
                  placeholder="Describe this special moment..."
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  rows="4"
                  required
                />
              </div>

              <div className="form-group">
                <label>Photos</label>
                <div className="photo-upload">
                  <label className="upload-btn">
                    <Camera size={20} />
                    <span>Add Photos</span>
                    <input
                      type="file"
                      accept="image/*"
                      multiple
                      onChange={handlePhotoUpload}
                      style={{ display: 'none' }}
                    />
                  </label>
                  {formData.photos.length > 0 && (
                    <div className="photo-preview">
                      {formData.photos.map((photo, index) => (
                        <div key={index} className="preview-item">
                          <img src={photo} alt={`Upload ${index + 1}`} />
                          <button
                            type="button"
                            className="remove-photo"
                            onClick={() => removePhoto(index)}
                          >
                            <X size={14} />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              <div className="modal-actions">
                <button type="submit" className="btn-primary">
                  <Save size={16} /> {editingMemory ? 'Save Changes' : 'Save Memory'}
                </button>
                <button type="button" className="btn-secondary" onClick={() => { setShowForm(false); resetForm(); }}>
                  <X size={16} /> Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      <div className="memories-grid">
        {sortedMemories.length === 0 ? (
          <div className="empty-state scrapbook-empty">
            <BookHeart size={48} />
            <h3>Your scrapbook is empty</h3>
            <p>Start capturing your honeymoon memories!</p>
            <button className="btn-primary" onClick={() => { resetForm(); setShowForm(true); }}>
              <Plus size={18} /> Add Your First Memory
            </button>
          </div>
        ) : (
          sortedMemories.map(memory => (
            <MemoryCard
              key={memory.id}
              memory={memory}
              onEdit={handleEdit}
              onDelete={deleteMemory}
            />
          ))
        )}
      </div>
    </div>
  );
}
