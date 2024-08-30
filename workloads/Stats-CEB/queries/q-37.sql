SELECT COUNT(*)
FROM comments AS c,
  postLinks AS pl,
  postHistory AS ph,
  votes AS v
WHERE pl.PostId = c.PostId
  AND c.PostId = ph.PostId
  AND ph.PostId = v.PostId
  AND ph.CreationDate >= CAST('2011-05-07 21:47:19' AS timestamp)
  AND ph.CreationDate <= CAST('2014-09-10 13:19:54' AS timestamp);
