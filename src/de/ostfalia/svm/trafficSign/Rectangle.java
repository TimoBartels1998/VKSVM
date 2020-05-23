package de.ostfalia.svm.trafficSign;
import java.nio.ByteBuffer;

import org.opencv.core.Rect;

/**
 * Rectangle
 */
public class Rectangle {
	public static final int BYTES = 16;
	
	public int x;
	public int y;
	public int radiusX;
	public int radiusY;
	
	/**
	 * Constructor
	 * @param minX Minimum x
	 * @param minY Minimum y
	 * @param maxX Maximum x
	 * @param maxY Maximum y
	 */
	public Rectangle(int x, int y, int radiusX, int radiusY) {
		this.x = x;
		this.y = y;
		this.radiusX = radiusX;
		this.radiusY = radiusY;
	}
	
	@Override
	public boolean equals(Object obj) {
		if(obj instanceof Rectangle) {
			Rectangle other = (Rectangle) obj;
			return (other.x == this.x) && (other.y == this.y) && (other.radiusX == this.radiusX)
					&& (other.radiusY == this.radiusY);
		} else {
			return false;
		}
	}
	
	/**
	 * Get Rect
	 * @return Rect
	 */
	public Rect getRect() {
		return new Rect(x, y, radiusX, radiusY);
	}
	
	/**
	 * Does other rectangle overlap this
	 * @param other Other rectangle
	 * @return True if overlaps, False else
	 */
	public boolean overlap(Rectangle other) {
		return isPointIn(other.x, other.y) || isPointIn(other.x, other.y + other.radiusY)
				|| isPointIn(other.x + other.radiusX, other.y)
				|| isPointIn(other.x + other.radiusX, other.y + other.radiusY);
	}
	
	/**
	 * Is point in rectangle
	 * @param x X
	 * @param y Y
	 * @return True if is in, False else
	 */
	private boolean isPointIn(int x, int y) {
		return (x > this.x) && (x < (this.x + this.radiusX)) && (y > this.y) && (y < (this.y + this.radiusY));
	}
	
	public static ByteBuffer RectangleToByteBuffer(Rectangle r) {
		ByteBuffer bb = ByteBuffer.allocate(16);
		bb.putInt(r.x);
		bb.putInt(r.y);
		bb.putInt(r.radiusX);
		bb.putInt(r.radiusY);
		return bb;
	}
	public static Rectangle ByteBufferToRectangle(ByteBuffer bb) {
		return new Rectangle(bb.getInt(0), bb.getInt(4), bb.get(8), bb.getInt(12));
	}
}
