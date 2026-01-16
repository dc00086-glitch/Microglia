import { useState } from 'react';
import { useHoneymoon } from '../context/HoneymoonContext';
import {
  Plane, Hotel, Ticket, Plus, Trash2, Edit2, Copy, Check, X, Save, DollarSign
} from 'lucide-react';
import { format, parseISO } from 'date-fns';

const bookingTypes = [
  { value: 'flight', label: 'Flight', icon: Plane },
  { value: 'hotel', label: 'Hotel', icon: Hotel },
  { value: 'activity', label: 'Activity', icon: Ticket },
  { value: 'restaurant', label: 'Restaurant', icon: Ticket },
  { value: 'transport', label: 'Transport', icon: Plane },
];

function BookingCard({ booking, onEdit, onDelete }) {
  const [copied, setCopied] = useState(false);
  const TypeIcon = bookingTypes.find(t => t.value === booking.type)?.icon || Ticket;

  const copyConfirmation = () => {
    navigator.clipboard.writeText(booking.confirmationNumber);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`booking-card booking-${booking.type}`}>
      <div className="booking-icon">
        <TypeIcon size={24} />
      </div>
      <div className="booking-content">
        <h3>{booking.title}</h3>
        <div className="booking-details">
          <p className="booking-date">
            {format(parseISO(booking.date), 'MMM d, yyyy')}
            {booking.time && ` at ${booking.time}`}
            {booking.checkOut && ` - ${format(parseISO(booking.checkOut), 'MMM d, yyyy')}`}
          </p>
          {booking.details && <p className="booking-info">{booking.details}</p>}
          <div className="confirmation-row">
            <span className="confirmation-label">Confirmation:</span>
            <code className="confirmation-number">{booking.confirmationNumber}</code>
            <button className="copy-btn" onClick={copyConfirmation}>
              {copied ? <Check size={14} /> : <Copy size={14} />}
            </button>
          </div>
        </div>
      </div>
      <div className="booking-cost">
        <DollarSign size={16} />
        <span>{booking.cost?.toLocaleString() || 0}</span>
      </div>
      <div className="booking-actions">
        <button className="icon-btn" onClick={() => onEdit(booking)}>
          <Edit2 size={16} />
        </button>
        <button className="icon-btn delete" onClick={() => onDelete(booking.id)}>
          <Trash2 size={16} />
        </button>
      </div>
    </div>
  );
}

export default function BookingsPage() {
  const { bookings, addBooking, updateBooking, deleteBooking, getTotalBudget } = useHoneymoon();
  const [showForm, setShowForm] = useState(false);
  const [editingBooking, setEditingBooking] = useState(null);
  const [filterType, setFilterType] = useState('all');
  const [formData, setFormData] = useState({
    type: 'flight',
    title: '',
    confirmationNumber: '',
    date: '',
    time: '',
    checkOut: '',
    details: '',
    cost: '',
  });

  const resetForm = () => {
    setFormData({
      type: 'flight',
      title: '',
      confirmationNumber: '',
      date: '',
      time: '',
      checkOut: '',
      details: '',
      cost: '',
    });
    setEditingBooking(null);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const bookingData = {
      ...formData,
      cost: parseFloat(formData.cost) || 0,
    };

    if (editingBooking) {
      updateBooking(editingBooking.id, bookingData);
    } else {
      addBooking(bookingData);
    }

    resetForm();
    setShowForm(false);
  };

  const handleEdit = (booking) => {
    setFormData({
      type: booking.type,
      title: booking.title,
      confirmationNumber: booking.confirmationNumber,
      date: booking.date,
      time: booking.time || '',
      checkOut: booking.checkOut || '',
      details: booking.details || '',
      cost: booking.cost?.toString() || '',
    });
    setEditingBooking(booking);
    setShowForm(true);
  };

  const filteredBookings = filterType === 'all'
    ? bookings
    : bookings.filter(b => b.type === filterType);

  const sortedBookings = [...filteredBookings].sort((a, b) =>
    new Date(a.date) - new Date(b.date)
  );

  return (
    <div className="bookings-page">
      <header className="page-header">
        <div className="header-content">
          <Plane size={32} />
          <div>
            <h1>Booking Manager</h1>
            <p>Track all your reservations and confirmations</p>
          </div>
        </div>
        <div className="header-stats">
          <div className="total-budget">
            <span className="budget-label">Total Spent</span>
            <span className="budget-value">${getTotalBudget().toLocaleString()}</span>
          </div>
          <button className="btn-primary" onClick={() => { resetForm(); setShowForm(true); }}>
            <Plus size={18} /> Add Booking
          </button>
        </div>
      </header>

      <div className="filter-bar">
        <button
          className={`filter-btn ${filterType === 'all' ? 'active' : ''}`}
          onClick={() => setFilterType('all')}
        >
          All ({bookings.length})
        </button>
        {bookingTypes.map(type => {
          const count = bookings.filter(b => b.type === type.value).length;
          return (
            <button
              key={type.value}
              className={`filter-btn ${filterType === type.value ? 'active' : ''}`}
              onClick={() => setFilterType(type.value)}
            >
              <type.icon size={14} />
              {type.label} ({count})
            </button>
          );
        })}
      </div>

      {showForm && (
        <div className="modal-overlay" onClick={() => { setShowForm(false); resetForm(); }}>
          <div className="modal booking-modal" onClick={(e) => e.stopPropagation()}>
            <h2>{editingBooking ? 'Edit Booking' : 'Add New Booking'}</h2>
            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label>Booking Type</label>
                <div className="type-selector">
                  {bookingTypes.map(type => (
                    <button
                      key={type.value}
                      type="button"
                      className={`type-btn ${formData.type === type.value ? 'active' : ''}`}
                      onClick={() => setFormData({ ...formData, type: type.value })}
                    >
                      <type.icon size={18} />
                      <span>{type.label}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="form-group">
                <label>Title</label>
                <input
                  type="text"
                  placeholder="e.g., Flight to Rome"
                  value={formData.title}
                  onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                  required
                />
              </div>

              <div className="form-group">
                <label>Confirmation Number</label>
                <input
                  type="text"
                  placeholder="ABC123XYZ"
                  value={formData.confirmationNumber}
                  onChange={(e) => setFormData({ ...formData, confirmationNumber: e.target.value })}
                  required
                />
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label>{formData.type === 'hotel' ? 'Check-in Date' : 'Date'}</label>
                  <input
                    type="date"
                    value={formData.date}
                    onChange={(e) => setFormData({ ...formData, date: e.target.value })}
                    required
                  />
                </div>
                {formData.type === 'hotel' ? (
                  <div className="form-group">
                    <label>Check-out Date</label>
                    <input
                      type="date"
                      value={formData.checkOut}
                      onChange={(e) => setFormData({ ...formData, checkOut: e.target.value })}
                    />
                  </div>
                ) : (
                  <div className="form-group">
                    <label>Time</label>
                    <input
                      type="time"
                      value={formData.time}
                      onChange={(e) => setFormData({ ...formData, time: e.target.value })}
                    />
                  </div>
                )}
              </div>

              <div className="form-group">
                <label>Details</label>
                <textarea
                  placeholder="Additional details..."
                  value={formData.details}
                  onChange={(e) => setFormData({ ...formData, details: e.target.value })}
                  rows="2"
                />
              </div>

              <div className="form-group">
                <label>Cost ($)</label>
                <input
                  type="number"
                  placeholder="0.00"
                  value={formData.cost}
                  onChange={(e) => setFormData({ ...formData, cost: e.target.value })}
                  min="0"
                  step="0.01"
                />
              </div>

              <div className="modal-actions">
                <button type="submit" className="btn-primary">
                  <Save size={16} /> {editingBooking ? 'Save Changes' : 'Add Booking'}
                </button>
                <button type="button" className="btn-secondary" onClick={() => { setShowForm(false); resetForm(); }}>
                  <X size={16} /> Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      <div className="bookings-list">
        {sortedBookings.length === 0 ? (
          <div className="empty-state">
            <Plane size={48} />
            <h3>No bookings yet</h3>
            <p>Add your flights, hotels, and activities!</p>
            <button className="btn-primary" onClick={() => { resetForm(); setShowForm(true); }}>
              <Plus size={18} /> Add Your First Booking
            </button>
          </div>
        ) : (
          sortedBookings.map(booking => (
            <BookingCard
              key={booking.id}
              booking={booking}
              onEdit={handleEdit}
              onDelete={deleteBooking}
            />
          ))
        )}
      </div>
    </div>
  );
}
