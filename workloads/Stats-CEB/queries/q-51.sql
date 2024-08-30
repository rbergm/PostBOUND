SELECT COUNT(*)
FROM comments AS c,
  posts AS p,
  postHistory AS ph,
  badges AS b,
  users AS u
WHERE u.Id = c.UserId
  AND u.Id = p.OwnerUserId
  AND u.Id = ph.UserId
  AND u.Id = b.UserId
  AND c.CreationDate >= CAST('2010-07-31 05:18:59' AS timestamp)
  AND c.CreationDate <= CAST('2014-09-12 07:59:13' AS timestamp)
  AND p.Score >= -2
  AND p.ViewCount >= 0
  AND p.ViewCount <= 18281
  AND ph.PostHistoryTypeId = 2
  AND b.Date >= CAST('2010-10-20 08:33:44' AS timestamp)
  AND u.Views >= 0
  AND u.Views <= 75;
